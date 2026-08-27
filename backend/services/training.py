"""
Training run lifecycle: start, monitor, stop, and record history.

Training happens in a separate process. The web process never imports torch or
ultralytics — it only writes a config file, launches a worker, and reads the
status the worker writes back. That keeps a crashed or out-of-memory training
run from taking the whole server down with it.
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from config import BACKEND_ROOT, PROJECTS_ROOT, WEIGHTS_CACHE_DIR
from services import augment as augment_service
from services import dataset, projects, trainaug
from services.atomicio import read_json as _read_json_atomic, write_json
from services.projects import ProjectError

# Model identifier -> pretrained checkpoint. Everything here goes through the
# ultralytics trainer; faster_rcnn is handled by the separate torchvision worker.
YOLO_WEIGHTS = {
    'yolov8n': 'yolov8n.pt', 'yolov8s': 'yolov8s.pt', 'yolov8m': 'yolov8m.pt',
    'yolov8l': 'yolov8l.pt', 'yolov8x': 'yolov8x.pt',
    'yolov9t': 'yolov9t.pt', 'yolov9s': 'yolov9s.pt', 'yolov9m': 'yolov9m.pt',
    'yolov9c': 'yolov9c.pt', 'yolov9e': 'yolov9e.pt',
    'yolov10n': 'yolov10n.pt', 'yolov10s': 'yolov10s.pt', 'yolov10m': 'yolov10m.pt',
    'yolov10l': 'yolov10l.pt', 'yolov10x': 'yolov10x.pt',
    'yolo11n': 'yolo11n.pt', 'yolo11s': 'yolo11s.pt', 'yolo11m': 'yolo11m.pt',
    'yolo11l': 'yolo11l.pt', 'yolo11x': 'yolo11x.pt',
    'rtdetr-l': 'rtdetr-l.pt', 'rtdetr-x': 'rtdetr-x.pt',
}
FRCNN_TYPE = 'faster_rcnn'
MODEL_TYPES = sorted(YOLO_WEIGHTS) + [FRCNN_TYPE]

YOLO_EXPORT_FORMATS = {'pt', 'onnx', 'torchscript', 'openvino', 'engine', 'tflite', 'blob'}
# No 'pt' for Faster R-CNN: a torch.save(model) pickle needs this exact
# source tree to load and cannot be tested in the model tester.
FRCNN_EXPORT_FORMATS = {'onnx', 'torchscript', 'blob'}

# Statuses that mean a worker process should still be alive. 'stopping' is
# included: the worker is asked to finish the current epoch, so it is still
# running until it writes its own final status.
ACTIVE_STATUSES = ('preparing', 'running', 'stopping')

# One lock per project. Without it two simultaneous /training/start calls both
# pass the "already running?" check, then race inside build_yolo_dataset() —
# which begins by deleting the dataset directory the other one is writing into
# — and both spawn a worker, of which only the second pid is recorded, leaving
# the first unstoppable.
_project_locks = {}
_locks_guard = threading.Lock()

# How long to let a worker finish the epoch it is on after a stop request.
GRACEFUL_STOP_SECONDS = 90


def _project_lock(name):
    with _locks_guard:
        if name not in _project_locks:
            _project_locks[name] = threading.Lock()
        return _project_locks[name]


def _weights_path(model_type):
    """
    Absolute path for a pretrained checkpoint.

    ultralytics downloads a bare file name into the current working directory.
    Handing it an absolute path inside data/weights keeps the download in one
    known place and lets later runs reuse it instead of re-fetching.
    """
    filename = YOLO_WEIGHTS.get(model_type)
    if not filename:
        return None
    WEIGHTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(WEIGHTS_CACHE_DIR / filename)


def config_path(name):
    return projects.training_dir(name) / 'training_config.json'


def log_path(name):
    return projects.training_dir(name) / 'training.log'


def history_path(name):
    return projects.training_dir(name) / 'history.json'


def _read_json(path, default=None):
    return _read_json_atomic(path, default)


def _write_json(path, data):
    write_json(path, data)


def _process_handle(pid, created_at=None):
    """
    The psutil handle for our worker, or None.

    A pid on its own is not an identity: the OS reuses them, and a worker that
    died without writing a final status (an OOM kill, a machine restart) can
    leave a pid behind that now belongs to something else entirely. Matching
    the recorded creation time as well makes sure a stop request can never
    terminate an unrelated process and its children.
    """
    if not pid:
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        proc = psutil.Process(int(pid))
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return None
        if created_at is not None and abs(proc.create_time() - float(created_at)) > 1.0:
            return None  # same pid, different process
        return proc
    except Exception:  # noqa: BLE001 - any lookup failure means "not our process"
        return None


def _process_alive(pid, created_at=None):
    """True when the recorded worker process is still running."""
    if not pid:
        return False
    try:
        import psutil  # noqa: F401
    except ImportError:
        # Without psutil liveness cannot be determined; assume alive rather
        # than falsely reporting a crash.
        return True
    return _process_handle(pid, created_at) is not None


# ── status ──────────────────────────────────────────────────────────────────

def get_status(name):
    """
    Current run state, with dead workers reported as crashed.

    A worker killed by the OS (out of memory, machine restart) never gets to
    write a final status, which used to leave the UI stuck on "running"
    forever. Checking the recorded pid turns that into an honest failure.
    """
    projects.get_project(name)
    status = _read_json(config_path(name))
    if not status:
        return None

    if (status.get('status') in ACTIVE_STATUSES
            and not _process_alive(status.get('pid'), status.get('pid_created_at'))
            # 'preparing' means start() has written the config but not yet
            # spawned the worker, so there is legitimately no pid to find.
            # Treating that as a crash produced a spurious "stopped
            # unexpectedly" on every poll that landed mid-launch.
            and not (status.get('status') == 'preparing' and status.get('pid') is None)):
        if status.get('status') == 'stopping':
            # The user asked it to stop and the process is gone: that is a
            # completed stop, not a crash.
            status['status'] = 'stopped'
        else:
            status['status'] = 'failed'
            status['error'] = status.get('error') or (
                'The training process stopped unexpectedly. Check the log below — '
                'the most common cause is running out of GPU or system memory.'
            )
        status['completed_at'] = status.get('completed_at') or datetime.now().isoformat()
        status['pid'] = None
        _write_json(config_path(name), status)

    return status


def get_logs(name, last_n=200):
    projects.get_project(name)
    path = log_path(name)
    if not path.exists():
        return []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        return []
    return [line.rstrip('\n') for line in lines[-max(1, int(last_n)):]]


def reset_status(name):
    """Force a stuck run back to idle without touching a live process."""
    projects.get_project(name)
    status = _read_json(config_path(name))
    if not status:
        return {'message': 'No training run to reset'}
    if (status.get('status') in ACTIVE_STATUSES
            and _process_alive(status.get('pid'), status.get('pid_created_at'))):
        raise ProjectError(
            'Training is still running. Stop it first, then reset.'
        )
    previous = status.get('status', 'unknown')
    status['status'] = 'idle'
    status['error'] = None
    status['pid'] = None
    status['pid_created_at'] = None
    _write_json(config_path(name), status)
    return {'message': f'Status reset from {previous} to idle'}


def stop(name):
    """
    Ask the worker to stop, wait for it to finish the current epoch, then kill.

    The polite path matters: the worker checks this flag between epochs and
    exits through its normal completion code, which runs the exports and
    records the run in history.json. The previous version called terminate()
    with no grace period — on Windows that is TerminateProcess, so the worker
    never observed the request and none of that ever ran.
    """
    projects.get_project(name)
    status = _read_json(config_path(name))
    if not status:
        raise ProjectError('No training run found', status=404)
    if status.get('status') not in ACTIVE_STATUSES:
        raise ProjectError('Training is not running')

    status['status'] = 'stopping'
    status['stop_requested_at'] = datetime.now().isoformat()
    _write_json(config_path(name), status)

    pid = status.get('pid')
    created_at = status.get('pid_created_at')
    proc = _process_handle(pid, created_at)
    if proc is None:
        # Nothing left to signal; settle the recorded state and return.
        return _finalise_stop(name, killed=False)

    deadline = time.time() + GRACEFUL_STOP_SECONDS
    while time.time() < deadline:
        if not proc.is_running():
            # Exited on its own, so its own final status (with exports and a
            # history entry) is already on disk.
            current = _read_json(config_path(name)) or {}
            if current.get('status') not in ACTIVE_STATUSES + ('stopping',):
                return {'message': 'Training stopped', 'process_terminated': False}
            return _finalise_stop(name, killed=False)
        time.sleep(0.5)

    # It ignored the request — an epoch longer than the grace period, or a
    # hang inside a native call. Terminate, then kill the tree if needed.
    killed = False
    try:
        import psutil
        children = proc.children(recursive=True)
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except psutil.TimeoutExpired:
            for child in children:
                try:
                    child.kill()
                except Exception:  # noqa: BLE001
                    pass
            proc.kill()
        killed = True
    except ImportError:
        try:
            os.kill(int(pid), 9)
            killed = True
        except OSError:
            pass
    except Exception:  # noqa: BLE001 - already gone
        killed = True

    return _finalise_stop(name, killed=killed)


def _finalise_stop(name, killed):
    """Record a terminal 'stopped' state unless the worker already wrote one."""
    status = _read_json(config_path(name)) or {}
    if status.get('status') in ACTIVE_STATUSES:
        status['status'] = 'stopped'
        status['completed_at'] = datetime.now().isoformat()
        status['pid'] = None
        status['pid_created_at'] = None
        _write_json(config_path(name), status)
    return {'message': 'Training stopped', 'process_terminated': killed}


# ── start ───────────────────────────────────────────────────────────────────

def _validate_request(model_type, epochs, batch_size, img_size, learning_rate,
                      export_formats):
    if model_type not in MODEL_TYPES:
        raise ProjectError(f'Unknown model type "{model_type}"')

    try:
        epochs = int(epochs)
        batch_size = int(batch_size)
        img_size = int(img_size)
        learning_rate = float(learning_rate)
    except (TypeError, ValueError):
        raise ProjectError('epochs, batch_size, img_size and learning_rate must be numbers')

    if not 1 <= epochs <= 1000:
        raise ProjectError('epochs must be between 1 and 1000')
    if not 1 <= batch_size <= 128:
        raise ProjectError('batch_size must be between 1 and 128')
    if not 64 <= img_size <= 2048:
        raise ProjectError('img_size must be between 64 and 2048')
    # YOLO requires a multiple of the maximum stride (32).
    if img_size % 32 != 0:
        raise ProjectError('img_size must be a multiple of 32 (e.g. 416, 640, 960)')
    if not 0 < learning_rate < 1:
        raise ProjectError('learning_rate must be between 0 and 1')

    allowed = FRCNN_EXPORT_FORMATS if model_type == FRCNN_TYPE else YOLO_EXPORT_FORMATS
    formats = [str(f).strip().lower() for f in (export_formats or []) if str(f).strip()]
    unknown = [f for f in formats if f not in allowed]
    if unknown:
        raise ProjectError(
            f'Unsupported export format(s) for {model_type}: {", ".join(unknown)}. '
            f'Supported: {", ".join(sorted(allowed))}'
        )
    # The trained weights themselves are always produced; for YOLO that is a
    # .pt, for Faster R-CNN a .pth written by the worker regardless.
    if model_type != FRCNN_TYPE and 'pt' not in formats:
        formats.insert(0, 'pt')

    return epochs, batch_size, img_size, learning_rate, formats


def start(name, model_type='yolo11s', epochs=100, batch_size=16, img_size=640,
          learning_rate=0.001, export_formats=None, model_name='',
          augmentation=None, generate_filters=None):
    """
    Prepare the dataset and launch a training worker in the background.

    `augmentation` overrides the per-epoch settings the trainer uses; anything
    not given falls back to what services/trainaug.py recommends from the
    project's own class names. `generate_filters` asks for filtered copies of
    the annotated images to be written first, and is a list of preset names or
    True for the default set.
    """
    projects.get_project(name)
    lock = _project_lock(name)
    # Non-blocking: a second start should be told the first one is already
    # under way, not queued behind a dataset rebuild that may take minutes.
    if not lock.acquire(blocking=False):
        raise ProjectError('A training run is already being started for this project')
    try:
        return _start_locked(name, model_type, epochs, batch_size, img_size,
                             learning_rate, export_formats, model_name,
                             augmentation, generate_filters)
    finally:
        lock.release()


def _start_locked(name, model_type, epochs, batch_size, img_size,
                  learning_rate, export_formats, model_name,
                  augmentation=None, generate_filters=None):

    epochs, batch_size, img_size, learning_rate, export_formats = _validate_request(
        model_type, epochs, batch_size, img_size, learning_rate, export_formats
    )

    current = get_status(name)
    if current and current.get('status') in ACTIVE_STATUSES:
        raise ProjectError('A training run is already in progress for this project')

    summary = projects.dataset_summary(name)
    if summary['annotated_images'] == 0:
        raise ProjectError('No annotated images. Annotate some images before training.')
    if summary['num_classes'] == 0:
        raise ProjectError('No classes defined. Tag some boxes before training.')

    training_dir = projects.training_dir(name)
    training_dir.mkdir(parents=True, exist_ok=True)
    results_dir = training_dir / 'runs'
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = (model_name or '').strip() or \
        f'{name}_{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    # Capped well below what the run directory can take: the name appears in
    # <project>/training/runs/<name>/weights/best.pt and again in <name>_val,
    # and Windows still stops at 260 characters unless long paths are enabled.
    model_name = projects.safe_filename(model_name.replace(' ', '_'), max_length=64)

    # Checked against both the run directory and the recorded history: a run
    # can finish, be archived in history.json, and still have its weights
    # sitting in runs/<name>/. Reusing the name would overwrite them while
    # history kept pointing at the path.
    previous_names = {entry.get('model_name') for entry in get_history(name)}
    if (results_dir / model_name).exists() or model_name in previous_names:
        raise ProjectError(
            f'A run called "{model_name}" already exists for this project. '
            'Choose a different name so its weights are not overwritten.'
        )

    # Filtered copies are written before the split is built, so the new images
    # are part of the dataset rather than a separate pass. Grouping by source
    # keeps every copy of one photograph on the same side of the train/val
    # line, so a variant of a training image can never appear in validation.
    filter_report = None
    if generate_filters:
        presets = (trainaug.DEFAULT_PRESETS if generate_filters is True
                   else [t for t in generate_filters
                         if t in augment_service.ALL_COLOR_TONES])
        if presets:
            filter_report = augment_service.augment_color_images(
                name, tones=presets, variants_per_tone=1,
                require_all_annotated=False)
            summary = projects.dataset_summary(name)

    report = dataset.build_yolo_dataset(name)

    # What the trainer will do to each image every epoch. Left to ultralytics'
    # own defaults this mirrors half of them, which is wrong for any class that
    # reads as text -- see services/trainaug.py.
    advice = trainaug.recommend(name)
    settings = dict(advice['settings'])
    settings.update(trainaug.sanitise(augmentation))

    config = {
        'project_name': name,
        'model_name': model_name,
        'augmentation': settings,
        'augmentation_reasons': advice['reasons'],
        'orientation_sensitive': advice['orientation_sensitive'],
        'generated_filters': (filter_report or {}).get('created_count', 0),
        'generated_filter_presets': (filter_report or {}).get('tones', []),
        'model_type': model_type,
        'weights': _weights_path(model_type),
        'epochs': epochs,
        'batch_size': batch_size,
        'img_size': img_size,
        'learning_rate': learning_rate,
        'export_formats': export_formats,
        'classes': report['classes'],
        'dataset_path': report['dataset_path'],
        'data_yaml': report['data_yaml'],
        'project_path': str(projects.project_dir(name)),
        'results_dir': str(results_dir.resolve()),
        'train_images': report['train_images'],
        'val_images': report['val_images'],
        'train_boxes': report['train_boxes'],
        'val_boxes': report['val_boxes'],
        'total_images': report['train_images'] + report['val_images'],
        'started_at': datetime.now().isoformat(),
        'completed_at': None,
        'status': 'preparing',
        'current_epoch': 0,
        'total_epochs': epochs,
        'metrics': {},
        'metrics_history': [],
        'best_model': None,
        'exported_models': {},
        'error': None,
        'pid': None,
        'pid_created_at': None,
    }
    _write_json(config_path(name), config)

    header = [
        f'[{datetime.now().isoformat()}] Training run "{model_name}"',
        f'Model: {model_type} | epochs={epochs} batch={batch_size} '
        f'imgsz={img_size} lr={learning_rate}',
        f'Classes ({len(report["classes"])}): {", ".join(report["classes"])}',
        f'Dataset: {report["train_images"]} train / {report["val_images"]} val images, '
        f'{report["train_boxes"]} / {report["val_boxes"]} boxes',
    ]
    if report['empty_classes']:
        header.append(f'WARNING: classes with no boxes: {", ".join(report["empty_classes"])}')
    if report['skipped_count']:
        header.append(f'WARNING: {report["skipped_count"]} images skipped during dataset build')
    header.append('')

    with open(log_path(name), 'w', encoding='utf-8') as f:
        f.write('\n'.join(header) + '\n')

    worker = (BACKEND_ROOT / 'training' /
              ('frcnn_worker.py' if model_type == FRCNN_TYPE else 'yolo_worker.py'))

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'
    env['TQDM_DISABLE'] = '1'
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    env['PYTHONPATH'] = os.pathsep.join(
        [str(BACKEND_ROOT), env.get('PYTHONPATH', '')]
    ).strip(os.pathsep)

    WEIGHTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path(name), 'a', encoding='utf-8', buffering=1)
    try:
        popen_kwargs = {}
        if sys.platform == 'win32':
            popen_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        else:
            popen_kwargs['start_new_session'] = True
        process = subprocess.Popen(
            [sys.executable, '-u', str(worker), str(config_path(name))],
            stdout=log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env, cwd=str(WEIGHTS_CACHE_DIR), **popen_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - report as a normal API failure
        log_handle.close()
        config['status'] = 'failed'
        config['error'] = f'Could not start the training process: {exc}'
        _write_json(config_path(name), config)
        raise ProjectError(config['error'], status=500)
    finally:
        # The child holds its own duplicated handle; this one is not needed.
        log_handle.close()

    # Merge into whatever is on disk rather than writing the stale local dict:
    # a fast-failing worker may already have recorded a real error here, and
    # overwriting it with 'running' would hide the reason the run died.
    created_at = None
    try:
        import psutil
        created_at = psutil.Process(process.pid).create_time()
    except Exception:  # noqa: BLE001 - identity check degrades to pid-only
        pass

    current = _read_json(config_path(name)) or config
    if current.get('status') == 'preparing':
        current['status'] = 'running'
    current['pid'] = process.pid
    current['pid_created_at'] = created_at
    _write_json(config_path(name), current)

    return {'message': 'Training started', 'config': current, 'dataset': report}


# ── results ─────────────────────────────────────────────────────────────────

def list_models(name):
    """Every weight file produced for this project, newest first."""
    projects.get_project(name)
    runs_dir = projects.training_dir(name) / 'runs'
    models = []
    if runs_dir.exists():
        for pattern in ('*.pt', '*.pth', '*.onnx', '*.torchscript', '*.blob'):
            for path in runs_dir.rglob(pattern):
                if not path.is_file():
                    continue
                stat = path.stat()
                models.append({
                    'name': path.name,
                    'run': path.relative_to(runs_dir).parts[0],
                    'path': str(path),
                    'format': path.suffix.lstrip('.'),
                    'size_mb': round(stat.st_size / 1024 / 1024, 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
    return sorted(models, key=lambda m: m['modified'], reverse=True)


def list_all_models():
    """
    Every weight file this installation has produced, across every project.

    Testing a model used to mean finding best.pt somewhere under
    data/projects/<name>/training/runs/<run>/weights and uploading it back to
    the server that had just written it. The application already knows where
    those files are, so it can offer them directly.
    """
    everything = []
    for project in projects.list_projects():
        name = project['name'] if isinstance(project, dict) else project
        try:
            found = list_models(name)
        except ProjectError:
            continue
        for model in found:
            everything.append({**model, 'project': name})
    return sorted(everything, key=lambda m: m['modified'], reverse=True)


DOWNLOADABLE_SUFFIXES = {'.pt', '.pth', '.onnx', '.torchscript', '.blob',
                         '.engine', '.tflite', '.yaml', '.csv'}


def resolve_trained_model(raw_path):
    """
    Turn a client-supplied path into a model file this installation produced.

    Same reasoning as resolve_model_path, widened from one project to the whole
    projects tree so a model can be tested from wherever it was trained. The
    containment check still uses resolved paths and relative_to rather than a
    string prefix, so a traversal cannot climb out, and the extension whitelist
    keeps the endpoint from loading anything that is not a model.
    """
    if not raw_path:
        raise ProjectError('A model path is required')
    candidate = Path(raw_path).resolve()
    roots = [Path(PROJECTS_ROOT).resolve(),
             Path(WEIGHTS_CACHE_DIR).resolve()]
    if not any(_is_within(candidate, root) for root in roots):
        raise ProjectError('That model is not one this server trained', status=403)
    if candidate.suffix.lower() not in DOWNLOADABLE_SUFFIXES:
        raise ProjectError(
            f'Not a model file ({", ".join(sorted(DOWNLOADABLE_SUFFIXES))})',
            status=403)
    if not candidate.is_file():
        raise ProjectError('Model file not found', status=404)
    return candidate


def _is_within(candidate, root):
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_model_path(name, raw_path):
    """
    Turn a client-supplied model path into a real file inside this project.

    Two checks, both necessary. Comparing resolved paths with relative_to
    (rather than string prefixes) stops "..\..\etc\passwd" from escaping the
    project. The extension whitelist stops the endpoint from being a general
    file reader for everything else inside it — raw images, annotation files
    and training_config.json were all downloadable before.
    """
    projects.get_project(name)
    if not raw_path:
        raise ProjectError('A model path is required')
    candidate = Path(raw_path).resolve()
    base = projects.project_dir(name).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ProjectError('Model path is outside this project', status=403)
    if candidate.suffix.lower() not in DOWNLOADABLE_SUFFIXES:
        raise ProjectError(
            f'Only model files can be downloaded '
            f'({", ".join(sorted(DOWNLOADABLE_SUFFIXES))})', status=403
        )
    if not candidate.is_file():
        raise ProjectError('Model file not found', status=404)
    return candidate


def get_history(name=None):
    """Completed run records for one project, or for every project."""
    names = [name] if name else [p['name'] for p in projects.list_projects()]
    records = []
    for project_name in names:
        entries = _read_json(history_path(project_name), []) or []
        for entry in entries:
            entry.setdefault('project_name', project_name)
            records.append(entry)
    return sorted(records, key=lambda r: r.get('started_at') or '', reverse=True)
