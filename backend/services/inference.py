"""
Ad-hoc model testing: run an uploaded model over uploaded images.

Torch and ultralytics are imported lazily inside the functions that need them,
so importing this module (and therefore starting the server) stays fast and
does not require a working CUDA install.
"""

import base64
import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from config import PROJECTS_ROOT
from services.projects import ProjectError

SUPPORTED_SUFFIXES = {'.pt', '.pth', '.onnx', '.torchscript'}
MAX_IMAGES = 50


def _sha1(path):
    digest = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _labels_from_projects(upload_path):
    """
    Find the class list of a model that was trained by this app.

    Matching by size first and only then by hash keeps this cheap even with
    many projects, and means a user can drop in a downloaded model without
    having to retype its class names.
    """
    if not PROJECTS_ROOT.exists():
        return None, None

    upload_size = Path(upload_path).stat().st_size
    upload_hash = None

    for config_file in PROJECTS_ROOT.glob('*/training/training_config.json'):
        try:
            with open(config_file, 'r', encoding='utf-8-sig') as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        classes = config.get('classes') or []
        if not classes:
            continue

        candidates = {}
        if config.get('best_model'):
            candidates['best_model'] = config['best_model']
        for fmt, path in (config.get('exported_models') or {}).items():
            if path:
                candidates[fmt] = path

        for source, raw_path in candidates.items():
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = PROJECTS_ROOT.parent / candidate
            if not candidate.is_file():
                continue
            try:
                if candidate.stat().st_size != upload_size:
                    continue
                if upload_hash is None:
                    upload_hash = _sha1(upload_path)
                if _sha1(candidate) != upload_hash:
                    continue
            except OSError:
                continue
            return [str(c) for c in classes], {
                'project_name': config.get('project_name') or config_file.parent.parent.name,
                'source_key': source,
            }

    return None, None


def _draw_box(image, x1, y1, x2, y2, text):
    """Draw one detection with a per-label colour and a readable caption."""
    seed = int(hashlib.md5(text.split(' ')[0].encode('utf-8')).hexdigest()[:6], 16)
    colour = ((seed >> 16) & 0xFF, (seed >> 8) & 0xFF, seed & 0xFF)
    # Keep colours bright enough to see against dark images.
    colour = tuple(int(80 + c * 0.68) for c in colour)

    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    baseline = max(y1 - 1, text_h + 6)
    cv2.rectangle(image, (x1, baseline - text_h - 6),
                  (x1 + text_w + 10, baseline), colour, -1)
    luminance = 0.299 * colour[2] + 0.587 * colour[1] + 0.114 * colour[0]
    foreground = (10, 10, 10) if luminance > 140 else (245, 245, 245)
    cv2.putText(image, text, (x1 + 5, baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, foreground, 1, cv2.LINE_AA)


def _load_images(image_files):
    images = []
    for file in image_files[:MAX_IMAGES]:
        if not file or not file.filename:
            continue
        buffer = np.frombuffer(file.read(), np.uint8)
        if buffer.size == 0:
            continue
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            continue
        images.append((file.filename, decoded))
    return images


def _encode(image):
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    return 'data:image/jpeg;base64,' + base64.b64encode(encoded.tobytes()).decode('ascii')


def run_inference(model_file=None, image_files=None, score_threshold=0.5,
                  label_names=None, img_size=640, model_path=None):
    """
    Detect objects in each image and return annotated previews.

    The model arrives one of two ways. `model_file` is an upload, for weights
    that came from somewhere else. `model_path` names a file this server
    already has — the caller must have validated it against the projects tree
    first — so a model the application just trained can be tried without
    downloading it and posting it straight back.

    Returns a dict ready to be sent as JSON.
    """
    if model_path is not None:
        source_name = Path(model_path).name
    elif model_file is not None and model_file.filename:
        source_name = Path(model_file.filename).name
    else:
        raise ProjectError('Select a model file, or pick one this server trained')

    images = _load_images(image_files or [])
    if not images:
        raise ProjectError('Select at least one readable image')

    suffix = Path(source_name).suffix.lower()
    if suffix == '.blob':
        raise ProjectError(
            'A .blob is compiled for the OAK device\'s Myriad chip and cannot run '
            'on a CPU or GPU. Test the .onnx or .pt export instead.'
        )
    if suffix not in SUPPORTED_SUFFIXES:
        raise ProjectError(f'Supported model files: {", ".join(sorted(SUPPORTED_SUFFIXES))}')

    try:
        score_threshold = min(max(float(score_threshold), 0.0), 1.0)
    except (TypeError, ValueError):
        score_threshold = 0.5

    label_names = [name for name in (label_names or []) if name]
    label_source = 'manual' if label_names else 'unresolved'

    if model_path is not None:
        temp_path, is_upload = Path(model_path), False
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
        is_upload = True
    try:
        if is_upload:
            model_file.save(str(temp_path))

        if not label_names:
            resolved, meta = _labels_from_projects(temp_path)
            if resolved:
                label_names = resolved
                label_source = f'matched project "{meta["project_name"]}" ({meta["source_key"]})'

        # An ONNX export only accepts the resolution it was exported at, and a
        # .pt should default to what it was trained at rather than to the
        # screen's 640: running a 320-trained model at 640 detects far less,
        # with nothing to say why.
        from services import modelcache
        native, fixed = modelcache.native_input_size(temp_path)
        if native and (fixed or img_size in (None, 640)):
            img_size = native

        if suffix == '.pth':
            results, device_label, used_labels = _run_frcnn(
                temp_path, images, score_threshold, label_names, img_size
            )
            if not label_names and used_labels:
                label_names = used_labels
                label_source = 'generated from the checkpoint'
            model_format = 'pth'
        else:
            results, device_label, used_labels = _run_ultralytics(
                temp_path, images, score_threshold, label_names, img_size
            )
            if not label_names and used_labels:
                label_names = used_labels
                label_source = 'embedded in the model file'
            model_format = suffix.lstrip('.')

        return {
            'model_name': source_name,
            'model_format': model_format,
            'device': device_label,
            'score_threshold': score_threshold,
            'resolved_label_names': label_names,
            'resolved_label_source': label_source,
            'total_images': len(results),
            'img_size': img_size,
            'results': results,
        }
    finally:
        # Only the temporary copy of an upload is ours to remove; a model the
        # server trained must survive being tested.
        #
        # missing_ok only covers FileNotFoundError. On Windows the file can
        # still be held open by the library that loaded it, and letting that
        # PermissionError escape a finally block would replace a perfectly
        # good result with a 500.
        if is_upload:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _run_ultralytics(model_path, images, threshold, label_names, img_size):
    """Run a YOLO / RT-DETR model (.pt, .onnx, .torchscript). Classes are 0-based."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ProjectError(
            'ultralytics is not installed. Run: pip install ultralytics', status=500
        )

    model = YOLO(str(model_path))
    # Models trained by this app carry their class names; prefer them over
    # whatever the user typed only when the user typed nothing.
    embedded = None
    names = getattr(model, 'names', None)
    if isinstance(names, dict) and names:
        embedded = [str(names[key]) for key in sorted(names)]
    elif isinstance(names, (list, tuple)) and names:
        embedded = [str(n) for n in names]

    active_labels = label_names or embedded or []

    def label_for(class_id):
        if 0 <= class_id < len(active_labels):
            return str(active_labels[class_id])
        return f'class_{class_id}'

    results = []
    for filename, bgr in images:
        prediction = model.predict(bgr, conf=threshold, verbose=False, imgsz=img_size)[0]
        boxes = prediction.boxes
        annotated = bgr.copy()
        detections = []

        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            for box, score, class_id in zip(xyxy, scores, class_ids):
                x1, y1, x2, y2 = (int(v) for v in box.tolist())
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                name = label_for(int(class_id))
                _draw_box(annotated, x1, y1, x2, y2, f'{name} {float(score):.2f}')
                detections.append({
                    'label_id': int(class_id), 'label_name': name,
                    'score': round(float(score), 4), 'box': [x1, y1, x2, y2],
                })

        encoded = _encode(annotated)
        if encoded:
            ordered = sort_reading_order(detections)
            results.append({
                'filename': filename,
                'detection_count': len(ordered),
                'detections': ordered,
                'reading': reading_of(ordered),
                'annotated_image': encoded,
            })

    return results, 'ultralytics', embedded


def _run_frcnn(model_path, images, threshold, label_names, img_size=640):
    """
    Run a torchvision Faster R-CNN state_dict (.pth). Classes are 1-based.

    img_size configures the model's own resize transform and must match the
    value the model was trained with, or every object is presented at a
    different scale than the weights expect.
    """
    try:
        import torch
    except ImportError:
        raise ProjectError('PyTorch is not installed', status=500)

    from training.frcnn_lib import create_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(str(model_path), map_location=device, weights_only=True)

    class_weight = state_dict.get('roi_heads.box_predictor.cls_score.weight')
    if class_weight is None:
        raise ProjectError(
            'This .pth is not a Faster R-CNN state_dict produced by this app.'
        )
    num_classes = int(class_weight.shape[0]) - 1  # index 0 is background

    model = create_model(num_classes=num_classes, pretrained=False, img_size=img_size)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    active_labels = list(label_names or [f'class_{i}' for i in range(1, num_classes + 1)])

    def label_for(class_id):
        if class_id <= 0:
            return 'background'
        index = class_id - 1
        return str(active_labels[index]) if index < len(active_labels) else f'class_{class_id}'

    results = []
    for filename, bgr in images:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(device)
        with torch.no_grad():
            output = model([tensor])[0]

        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy().astype(int)

        annotated = bgr.copy()
        detections = []
        for box, score, class_id in zip(boxes, scores, labels):
            if float(score) < threshold:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            # A detector can emit a sliver at the image edge that rounds to
            # zero width or height. Drawing it produces an invisible marker
            # and it inflates the detection count, so it is dropped.
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            name = label_for(int(class_id))
            _draw_box(annotated, x1, y1, x2, y2, f'{name} {float(score):.2f}')
            detections.append({
                'label_id': int(class_id), 'label_name': name,
                'score': round(float(score), 4), 'box': [x1, y1, x2, y2],
            })

        encoded = _encode(annotated)
        if encoded:
            ordered = sort_reading_order(detections)
            results.append({
                'filename': filename,
                'detection_count': len(ordered),
                'detections': ordered,
                'reading': reading_of(ordered),
                'annotated_image': encoded,
            })

    return results, str(device), active_labels


# ── Live and per-frame detection ────────────────────────────────────────────
#
# The still-image path above returns an annotated JPEG, which is the right
# answer for a handful of pictures and the wrong one for a stream: encoding a
# whole frame as base64 several times a second wastes far more time and
# bandwidth than the prediction itself, and the browser already has the pixels.
# These return coordinates only and let the page draw over what it is already
# showing.

def detect_frame(model_path, image_bgr, score_threshold=0.5, label_names=None,
                 img_size=640):
    """
    Boxes for one frame, using a model kept in memory between calls.

    Returns (detections, labels_in_use). Each detection is the same shape the
    still-image path produces, so the drawing code on the page is shared.
    """
    from services import modelcache

    entry = modelcache.get(model_path, img_size=img_size)
    try:
        score_threshold = min(max(float(score_threshold), 0.0), 1.0)
    except (TypeError, ValueError):
        score_threshold = 0.5

    label_names = [name for name in (label_names or []) if name]

    if entry['kind'] == 'ultralytics':
        return _detect_ultralytics(entry, image_bgr, score_threshold, label_names,
                                   effective_img_size(entry, img_size))
    return _detect_frcnn(entry, image_bgr, score_threshold, label_names)


def effective_img_size(entry, requested):
    """
    The size to actually predict at.

    An ONNX export has its resolution compiled into the graph, so a request for
    anything else is not a preference to weigh but an error onnxruntime raises.
    A .pt records what it was trained at, which is a strong default: a model
    trained at 320 and run at 640 detects far less, and nothing on screen would
    explain why.
    """
    native = entry.get('native_imgsz')
    if not native:
        return requested
    if entry.get('fixed_imgsz'):
        return native
    # The still-image screen sends 640 whether or not the user chose it, so an
    # untouched default should not override what the model was trained at.
    return native if requested in (None, 640) else requested


def _detect_ultralytics(entry, bgr, threshold, label_names, img_size):
    active = label_names or entry['labels'] or []

    def label_for(class_id):
        return str(active[class_id]) if 0 <= class_id < len(active) else f'class_{class_id}'

    prediction = entry['model'].predict(bgr, conf=threshold, verbose=False,
                                        imgsz=img_size)[0]
    boxes = prediction.boxes
    detections = []
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        for box, score, class_id in zip(xyxy, scores, class_ids):
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            detections.append({
                'label_id': int(class_id),
                'label_name': label_for(int(class_id)),
                'score': round(float(score), 4),
                'box': [x1, y1, x2, y2],
            })
    return sort_reading_order(detections), active


def _detect_frcnn(entry, bgr, threshold, label_names):
    import torch

    num_classes = entry['num_classes']
    active = list(label_names or [f'class_{i}' for i in range(1, num_classes + 1)])

    def label_for(class_id):
        if class_id <= 0:
            return 'background'
        index = class_id - 1
        return str(active[index]) if index < len(active) else f'class_{class_id}'

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    tensor = tensor.to(entry['device'])
    with torch.no_grad():
        output = entry['model']([tensor])[0]

    detections = []
    for box, score, class_id in zip(output['boxes'].cpu().numpy(),
                                    output['scores'].cpu().numpy(),
                                    output['labels'].cpu().numpy().astype(int)):
        if float(score) < threshold:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.tolist())
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        detections.append({
            'label_id': int(class_id),
            'label_name': label_for(int(class_id)),
            'score': round(float(score), 4),
            'box': [x1, y1, x2, y2],
        })
    return sort_reading_order(detections), active


def decode_frame(file_storage):
    """One uploaded frame as a BGR array, or None if it is not an image."""
    if file_storage is None:
        return None
    data = np.frombuffer(file_storage.read(), np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ── Reading order ───────────────────────────────────────────────────────────
#
# A detector returns boxes in whatever order it found them, which for most
# models is by confidence. That is the wrong order for anything being read
# rather than counted: a gauge showing 250 comes back as 0, 2, 5 or 5, 0, 2
# depending on which digit the model was surest about, and the number is lost.
#
# Sorting purely by x is right for a single line and wrong the moment there are
# two, because a digit low on the left would come before one high on the right.
# So boxes are grouped into lines first, by whether they overlap vertically,
# and only then read left to right within each line.

def _vertical_overlap(a, b):
    """How much of the shorter box's height the two share, 0 to 1."""
    top, bottom = max(a[1], b[1]), min(a[3], b[3])
    shared = max(0, bottom - top)
    shortest = min(a[3] - a[1], b[3] - b[1])
    return shared / shortest if shortest > 0 else 0.0


def sort_reading_order(detections, line_overlap=0.5):
    """
    Detections in the order a person would read them.

    Returns a new list; each entry gains `line` (0-based, top to bottom) and
    `position` (0-based within its line) so the caller can group them without
    repeating this work.

    `line_overlap` is how much of their height two boxes must share to count as
    being on the same line. Half is forgiving enough for digits that sit
    slightly high or low, and strict enough to separate two rows.
    """
    remaining = sorted(detections or [], key=lambda d: d['box'][1])
    lines = []
    for detection in remaining:
        for line in lines:
            # Compared against the line's first box rather than its last, so a
            # line cannot drift downwards across a long row.
            if _vertical_overlap(detection['box'], line[0]['box']) >= line_overlap:
                line.append(detection)
                break
        else:
            lines.append([detection])

    ordered = []
    for line_index, line in enumerate(lines):
        line.sort(key=lambda d: d['box'][0])
        for position, detection in enumerate(line):
            ordered.append({**detection, 'line': line_index, 'position': position})
    return ordered


def reading_of(detections):
    """
    The labels of these detections as one string, in reading order.

    For a project whose classes are the characters being read this is the
    answer the user actually wanted: '250', not a list of three boxes. Lines
    are separated by a newline so a two-row display stays two rows.
    """
    ordered = sort_reading_order(detections)
    lines = {}
    for detection in ordered:
        lines.setdefault(detection['line'], []).append(str(detection['label_name']))
    return '\n'.join(''.join(lines[index]) for index in sorted(lines))
