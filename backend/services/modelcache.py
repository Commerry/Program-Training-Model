"""
Keep loaded models in memory between requests.

Testing a still image loads the weights, runs one prediction and throws the
model away, which is fine when the whole exchange happens once. A webcam feed
asks for a prediction several times a second, and a video asks for hundreds in
a row: measured here, loading a small YOLO costs 131 ms against 40 ms to
actually predict a frame, so reloading each time caps the feed at about 6
frames per second when the same model held in memory reaches 25.

The cache is keyed by the file's path, size and modification time, so
retraining a model into the same filename invalidates the old entry rather than
quietly serving predictions from the previous run's weights.

Entries are large — hundreds of megabytes for a mid-sized model — so only a few
are kept and the least recently used is dropped first.
"""

import threading
from collections import OrderedDict
from pathlib import Path

from services.projects import ProjectError

# Two is enough for the normal case of comparing one model against another
# while leaving room on a 4 GB card.
MAX_CACHED_MODELS = 2

_lock = threading.Lock()
_cache = OrderedDict()


def _fingerprint(path):
    stat = Path(path).stat()
    return (str(Path(path).resolve()), stat.st_size, int(stat.st_mtime_ns))


def native_input_size(path):
    """
    The input size a model file was exported or trained at, if it records one.

    Returns (size, fixed). `fixed` means the model will accept nothing else.

    This matters most for ONNX, where the resolution is baked into the graph at
    export time. Feeding a 320-export a 640 frame does not merely predict
    badly, it raises "Got invalid dimensions for input: images ... Got: 640
    Expected: 320" from onnxruntime, which reaches the browser as an opaque
    500. A .pt records the size it was trained at as a hint, and honouring that
    saves the user from having to remember it: a model trained at 320 and run
    at 640 detects far less, with nothing on screen to say why.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.onnx':
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            shape = session.get_inputs()[0].shape
            # [batch, channels, height, width]; a dynamic axis is a string.
            if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
                return int(shape[2]), True
        except Exception:  # noqa: BLE001 - a missing size is not fatal
            pass
        return None, False

    try:
        from ultralytics import YOLO
        args = getattr(getattr(YOLO(str(path)), 'model', None), 'args', None)
        size = (args or {}).get('imgsz')
        if isinstance(size, (list, tuple)) and size:
            size = size[0]
        if isinstance(size, (int, float)) and size > 0:
            return int(size), False
    except Exception:  # noqa: BLE001
        pass
    return None, False


def _load_ultralytics(path):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ProjectError('ultralytics is not installed. Run: pip install ultralytics',
                           status=500)
    try:
        model = YOLO(str(path))
        names = getattr(model, 'names', None)
    except Exception as exc:  # noqa: BLE001 - the live paths need this too
        from services.inference import _loading_failed
        _loading_failed(path, exc)
    if isinstance(names, dict) and names:
        labels = [str(names[key]) for key in sorted(names)]
    elif isinstance(names, (list, tuple)) and names:
        labels = [str(n) for n in names]
    else:
        labels = []

    size, fixed = native_input_size(path)
    return {'kind': 'ultralytics', 'model': model, 'labels': labels, 'device': None,
            'native_imgsz': size, 'fixed_imgsz': fixed}


def _load_frcnn(path, img_size):
    try:
        import torch
    except ImportError:
        raise ProjectError('PyTorch is not installed', status=500)
    from training.frcnn_lib import create_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(str(path), map_location=device, weights_only=True)

    class_weight = state_dict.get('roi_heads.box_predictor.cls_score.weight')
    if class_weight is None:
        raise ProjectError('This .pth is not a Faster R-CNN state_dict produced by '
                           'this app.')
    num_classes = int(class_weight.shape[0]) - 1  # index 0 is background

    model = create_model(num_classes=num_classes, pretrained=False, img_size=img_size)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return {'kind': 'frcnn', 'model': model, 'labels': [], 'device': device,
            'num_classes': num_classes, 'native_imgsz': img_size,
            'fixed_imgsz': False}


def get(path, img_size=640):
    """
    A loaded model for this file, from memory when possible.

    Returns a dict describing it rather than the bare model, because the two
    families differ in ways every caller needs to know: ultralytics carries its
    own class names and takes BGR arrays, torchvision needs a device and counts
    classes from 1.
    """
    path = Path(path)
    if not path.is_file():
        raise ProjectError('Model file not found', status=404)

    suffix = path.suffix.lower()
    # img_size is part of the key for Faster R-CNN only, where it configures the
    # model's own resize transform and so changes what gets built. Ultralytics
    # takes it per prediction.
    key = _fingerprint(path) + ((img_size,) if suffix == '.pth' else ())

    with _lock:
        entry = _cache.get(key)
        if entry is not None:
            _cache.move_to_end(key)
            return entry

    # Loading happens outside the lock: it takes long enough that holding one
    # would serialise every request, including those for other models.
    entry = _load_frcnn(path, img_size) if suffix == '.pth' else _load_ultralytics(path)

    with _lock:
        _cache[key] = entry
        _cache.move_to_end(key)
        while len(_cache) > MAX_CACHED_MODELS:
            _cache.popitem(last=False)
    return entry


def clear():
    """Drop everything. Used by tests, and after deleting a model."""
    with _lock:
        _cache.clear()


def describe():
    """What is currently held, for diagnostics."""
    with _lock:
        return [{'path': key[0], 'kind': entry['kind'],
                 'classes': len(entry['labels']),
                 'imgsz': entry.get('native_imgsz')}
                for key, entry in _cache.items()]
