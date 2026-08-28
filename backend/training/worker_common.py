"""
Shared helpers for the training worker processes.

Workers communicate with the web process through two files only: the run's
training_config.json (status, progress, metrics) and training.log (human
readable output). Nothing is shared in memory, so a worker can be killed at any
point without corrupting server state.
"""

import itertools
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

_temp_counter = itertools.count()


def _atomic_write_json(path, data):
    """
    Write JSON via a uniquely named temp file, then rename.

    The unique name matters: the API process refreshes project stats while this
    worker updates its status, and a shared '<name>.tmp' lets one writer rename
    the file out from under the other.
    """
    path = Path(path)
    token = f'{os.getpid()}.{threading.get_ident()}.{next(_temp_counter)}'
    tmp = path.with_name(f'{path.name}.{token}.tmp')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def load_config(argv):
    if len(argv) < 2:
        print('Usage: <worker>.py <training_config.json>', file=sys.stderr)
        raise SystemExit(2)
    config_file = Path(argv[1])
    if not config_file.exists():
        print(f'Config file not found: {config_file}', file=sys.stderr)
        raise SystemExit(2)
    with open(config_file, 'r', encoding='utf-8-sig') as f:
        return config_file, json.load(f)


def read_config(config_file):
    try:
        with open(config_file, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def update_status(config_file, updates):
    """
    Merge updates into the run config.

    Written to a temp file and renamed so the web process, which polls this
    file every few seconds, never reads a half-written JSON document.
    """
    config_file = Path(config_file)
    config = read_config(config_file)
    config.update(updates)
    _atomic_write_json(config_file, config)
    return config


def stop_requested(config_file):
    return read_config(config_file).get('status') in ('stopping', 'stopped')


def make_logger(log_file):
    log_file = Path(log_file)

    def log(message):
        line = f'[{datetime.now().strftime("%H:%M:%S")}] {message}'
        # Only printed. The parent redirects this process's stdout into the
        # very same log file, so also writing it directly produced every line
        # twice, interleaved unpredictably between two append handles.
        print(line, flush=True)

    return log


def append_history(project_path, record):
    """Append one finished run to the project's history.json."""
    history_file = Path(project_path) / 'training' / 'history.json'
    history_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(history_file, 'r', encoding='utf-8-sig') as f:
            history = json.load(f)
        if not isinstance(history, list):
            history = []
    except (OSError, json.JSONDecodeError):
        history = []
    history.append(record)
    _atomic_write_json(history_file, history[-200:])


def quiet_environment():
    """Silence progress bars and warnings that break piped stdout on Windows."""
    import warnings
    warnings.filterwarnings('ignore')
    os.environ.setdefault('PYTHONWARNINGS', 'ignore')
    os.environ['TQDM_DISABLE'] = '1'
    os.environ.setdefault('YOLO_VERBOSE', 'False')


def describe_device():
    """Human-readable description of the device training will run on."""
    try:
        import torch
    except ImportError:
        return 'cpu', 'PyTorch is not installed'
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / 1e9
        return str(index), f'CUDA:{index} {name} ({total:.1f} GB)'
    return 'cpu', 'CPU (no CUDA device found — training will be slow)'


def self_check(model_path, dataset_path, img_size, log, sample=8, threshold=0.25,
               predict=None):
    """
    Ask the finished model about images from its own validation split.

    A run can report an excellent mAP and still produce weights that detect
    nothing. It happened here: turning on the dataloader's in-memory cache gave
    two runs the same mAP50 of 0.995, and one of them then scored 0.01 on every
    held-out image. The metric that should have warned about it read as
    perfect, so nothing downstream had any reason to doubt the weights until
    somebody tried them by hand.

    This is the cheapest possible guard against that whole class of failure:
    load what was just written, point it at pictures it was scored on, and see
    whether anything comes back. It proves nothing about accuracy -- a model
    that detects is not necessarily a good one -- but a model that detects
    nothing on the very images it was validated against is not usable, whatever
    the numbers say.

    `predict` takes an image path and returns the confidence scores it found,
    so each trainer can supply its own loader: ultralytics and torchvision have
    nothing in common here. Omitted, an ultralytics model is assumed.

    Returns a dict recorded with the run, or None if the check could not run.
    """
    from pathlib import Path

    val_dir = Path(dataset_path) / 'images' / 'val'
    if not val_dir.is_dir():
        return None
    images = sorted(p for p in val_dir.iterdir()
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
    if not images:
        return None
    images = images[:max(1, int(sample))]

    try:
        if predict is None:
            from ultralytics import YOLO
            model = YOLO(str(model_path))

            def predict(path):
                result = model.predict(str(path), imgsz=img_size, conf=threshold,
                                       verbose=False)[0]
                boxes = result.boxes
                if boxes is None or not len(boxes):
                    return []
                return [float(v) for v in boxes.conf]

        found_on = 0
        total = 0
        best = 0.0
        for path in images:
            scores = [s for s in (predict(path) or []) if s >= threshold]
            if scores:
                found_on += 1
                total += len(scores)
                best = max(best, max(scores))
    except Exception as exc:  # noqa: BLE001 - a failed check must not lose the run
        log(f'Self-check could not run (the weights are still there): {exc}')
        return None

    report = {
        'images_checked': len(images),
        'images_with_detections': found_on,
        'detections': total,
        'best_score': round(best, 4),
        'threshold': threshold,
        'usable': found_on > 0,
    }
    if found_on:
        log(f'Self-check: found {total} object(s) on {found_on}/{len(images)} '
            f'validation images (best {best:.2f}).')
    else:
        log(f'Self-check: found NOTHING on {len(images)} validation images at '
            f'{threshold:.2f}. These weights will not detect anything as they '
            'are, whatever the metrics above say.')
    return report
