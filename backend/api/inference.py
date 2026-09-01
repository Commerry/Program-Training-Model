"""Model testing and cross-project reporting endpoints."""

import csv
import hashlib
import io
import tempfile
from datetime import datetime
from pathlib import Path

from flask import Blueprint, Response, request

from api import login_required_api, ok
from config import INSTANCE_DIR
from services import inference, projects, report, training, videojob
from services.projects import ProjectError

inference_bp = Blueprint('inference', __name__)
inference_bp.before_request(login_required_api(lambda: None))


@inference_bp.post('/models/test')
def test_model():
    label_names = _label_names()
    img_size = _img_size()

    # Either an uploaded weights file, or the path of one this server trained.
    # The path is resolved against the projects tree before it reaches the
    # loader, so it cannot be pointed at an arbitrary file.
    raw_path = (request.form.get('model_path') or '').strip()
    model_path = training.resolve_trained_model(raw_path) if raw_path else None

    return ok(inference.run_inference(
        model_file=request.files.get('model'),
        model_path=model_path,
        image_files=request.files.getlist('images'),
        score_threshold=request.form.get('score_threshold', 0.5),
        label_names=label_names,
        img_size=img_size,
        # How an ONNX built elsewhere wants to be fed, once somebody has
        # worked it out with backend/tools/probe_onnx.py. Nothing in the file
        # records it, and guessing wrong returns confident nonsense.
        conventions=(request.form.get('onnx_conventions') or '').strip() or None,
    ))


@inference_bp.post('/models/import')
def import_model():
    """
    Bring in a detector built somewhere else, so it can pre-label a project.

    Pre-labelling is worth most on a project with nothing in it, which is
    exactly when this installation has no model of its own to offer. The file
    is kept in its own folder with the class names and, when they are known,
    the conventions it wants to be fed with -- an ONNX carries neither.
    """
    from services import imported
    return ok(imported.add(
        request.files.get('model'),
        name=(request.form.get('name') or '').strip() or None,
        labels_file=request.files.get('labels_file'),
        conventions=(request.form.get('onnx_conventions') or '').strip() or None,
    ))


@inference_bp.get('/models/imported')
def list_imported_models():
    from services import imported
    return ok({'models': imported.list_models()})


@inference_bp.post('/models/imported/<folder_name>/labels')
def set_imported_labels(folder_name):
    """Attach the class names to a model that was imported without them."""
    from services import imported
    return ok(imported.set_labels(folder_name, request.files.get('labels_file')))


@inference_bp.delete('/models/imported/<folder_name>')
def delete_imported_model(folder_name):
    from services import imported
    return ok(imported.remove(folder_name))


def _staged_model(upload):
    """
    Keep an uploaded model on disk under a name derived from its contents.

    A webcam feed asks for a prediction several times a second, and the loaded
    model is cached by file path, size and modification time. Writing each
    upload to a fresh temporary file and deleting it would defeat that cache
    entirely -- every frame would pay the load cost again -- and would leave
    cache entries pointing at files that no longer exist. Naming the file after
    the hash of its bytes means the same weights land on the same path every
    time, so the second frame of a session is a cache hit, and different
    weights can never collide on one name.
    """
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in inference.SUPPORTED_SUFFIXES:
        raise ProjectError('Supported model files: '
                           f'{", ".join(sorted(inference.SUPPORTED_SUFFIXES))}')

    staging = Path(INSTANCE_DIR) / 'uploaded-models'
    staging.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1()
    with tempfile.NamedTemporaryFile(dir=str(staging), delete=False,
                                     suffix=suffix) as tmp:
        temp_path = Path(tmp.name)
        while True:
            chunk = upload.stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            tmp.write(chunk)

    final = staging / f'{digest.hexdigest()}{suffix}'
    if final.exists():
        # Already staged by an earlier frame or an earlier session.
        temp_path.unlink(missing_ok=True)
    else:
        try:
            temp_path.replace(final)
        except OSError:
            # Another request staged the same weights between the check and
            # the rename; theirs is identical, so use it.
            temp_path.unlink(missing_ok=True)
            if not final.exists():
                raise

    _prune_staging(staging)
    return final


def _prune_staging(staging, keep=4):
    """Keep only the few most recently used uploads."""
    try:
        files = sorted((p for p in staging.iterdir() if p.is_file()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return
    for stale in files[keep:]:
        try:
            stale.unlink()
        except OSError:
            pass


def _model_for_request():
    """
    The model a request names, whether by path or by upload.

    A path is resolved against the projects tree before anything opens it, so
    it cannot be pointed at an arbitrary file.
    """
    raw_path = (request.form.get('model_path') or '').strip()
    if raw_path:
        return training.resolve_trained_model(raw_path)

    upload = request.files.get('model')
    if upload is None or not upload.filename:
        raise ProjectError('Select a model file, or pick one this server trained')
    return _staged_model(upload)


def _label_names():
    """
    The class names, typed in or uploaded.

    An ONNX carries no class names at all -- an export that has them keeps
    them in a labels.txt beside the model, which does not come along when only
    the model file is uploaded. That is why a Custom Vision model reported
    class_11 and class_19 rather than NonBad and TearTriangle. Uploading the
    file beats retyping twenty-one names in the right order.
    """
    text = (request.form.get('label_names') or '').strip()
    if not text:
        upload = request.files.get('labels_file')
        if upload and upload.filename:
            try:
                text = upload.read().decode('utf-8-sig', errors='replace')
            except OSError:
                text = ''
    return [part.strip() for part in text.replace('\n', ',').split(',') if part.strip()]


def _img_size():
    try:
        value = int(request.form.get('img_size') or 640)
    except (TypeError, ValueError):
        raise ProjectError('img_size must be a number')
    if not 64 <= value <= 4096:
        raise ProjectError('img_size must be between 64 and 4096')
    return value


@inference_bp.post('/models/detect')
def detect_frame():
    """
    Boxes for a single frame, for a webcam feed.

    Returns coordinates only. The page already has the pixels on screen, so
    sending an annotated copy back would cost more than the prediction does
    and would still arrive a frame late.
    """
    model_path = _model_for_request()
    frame = inference.decode_frame(request.files.get('frame')
                                   or request.files.get('image'))
    if frame is None:
        raise ProjectError('That frame could not be read as an image')

    detections, labels = inference.detect_frame(
        model_path, frame,
        score_threshold=request.form.get('score_threshold', 0.5),
        label_names=_label_names(),
        img_size=_img_size())
    return ok({
        'detections': detections,
        'detection_count': len(detections),
        # Already in reading order, and given as one string because for a
        # project whose classes are the characters on a display that is the
        # answer, not a list of boxes.
        'reading': inference.reading_of(detections),
        'label_names': labels,
        'width': int(frame.shape[1]),
        'height': int(frame.shape[0]),
    })


@inference_bp.post('/models/video')
def start_video():
    """Begin analysing an uploaded video. Returns straight away."""
    # An upload cannot be used here: the analysis outlives the request, and the
    # temporary file would be gone before the worker read it.
    raw_path = (request.form.get('model_path') or '').strip()
    if not raw_path:
        raise ProjectError('Pick a model this server trained to analyse a video')
    model_path = training.resolve_trained_model(raw_path)

    return ok({'job': videojob.start(
        model_path,
        request.files.get('video'),
        score_threshold=request.form.get('score_threshold', 0.5),
        label_names=_label_names(),
        img_size=_img_size(),
        sample_fps=request.form.get('sample_fps', videojob.DEFAULT_SAMPLE_FPS),
    )})


@inference_bp.get('/models/video/<job_id>')
def video_status(job_id):
    job = videojob.get(job_id)
    if job is None:
        raise ProjectError('No such video job', status=404)
    return ok({'job': job})


@inference_bp.post('/models/video/<job_id>/stop')
def video_stop(job_id):
    return ok({'job': videojob.stop(job_id)})


@inference_bp.post('/models/test/export')
def export_test_results():
    """
    A finished test written out as a spreadsheet, with the pictures in it.

    The results are posted back rather than looked up, because testing images
    is stateless -- there is no run to refer to. That means the annotated
    previews travel up as well as down, which is wasteful and is still the
    smaller cost: the alternative is holding every test in memory on the chance
    that somebody exports it.
    """
    data = request.get_json(silent=True) or {}
    results = data.get('results')
    if not isinstance(results, list):
        raise ProjectError('Nothing to export')

    workbook = report.build_workbook(results, meta={
        'model_name': data.get('model_name'),
        'device': data.get('device'),
        'score_threshold': data.get('score_threshold'),
    })

    stem = str(data.get('model_name') or 'model').rsplit('.', 1)[0]
    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    return Response(
        workbook,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition':
                 f'attachment; filename="{stem}_test_{stamp}.xlsx"'},
    )


@inference_bp.post('/models/video/<job_id>/csv')
@inference_bp.get('/models/video/<job_id>/csv')
def video_csv(job_id):
    """
    A finished video analysis as CSV, one row per detection.

    The page shows the boxes over the clip, which answers "did it work". It
    does not answer "what did it read at 12.4 seconds", and copying that out of
    a browser is not a thing anyone should have to do. Rows carry the reading
    of their frame as well as the individual box, so a spreadsheet can be
    filtered either way.
    """
    job = videojob.get(job_id)
    if job is None:
        raise ProjectError('No such video job', status=404)
    if job['status'] == 'running':
        raise ProjectError('That analysis is still running')

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(['time_s', 'frame', 'reading', 'label', 'score',
                     'x1', 'y1', 'x2', 'y2', 'line', 'position'])
    for entry in job.get('frames') or []:
        detections = entry.get('detections') or []
        if not detections:
            # A frame where nothing was found is a fact worth keeping: it is
            # the difference between "not looked at" and "looked, saw nothing".
            writer.writerow([entry.get('time_s'), entry.get('frame'),
                             '', '', '', '', '', '', '', '', ''])
            continue
        for detection in detections:
            box = detection.get('box') or [None] * 4
            writer.writerow([
                entry.get('time_s'), entry.get('frame'), entry.get('reading', ''),
                detection.get('label_name'), detection.get('score'),
                *box,
                detection.get('line'), detection.get('position'),
            ])

    filename = Path(job.get('filename') or 'detections').stem + '_detections.csv'
    return Response(
        buffer.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@inference_bp.get('/models')
def all_models():
    """Every model this installation has trained, newest first."""
    return ok({'models': training.list_all_models()})


@inference_bp.get('/history')
def all_history():
    """Every finished training run across every project, newest first."""
    return ok({'history': training.get_history()})


@inference_bp.get('/overview')
def overview():
    """
    Aggregate numbers for the dashboard and analytics pages.

    Computed from the same files the project pages read, so the dashboard can
    never disagree with the project it links to.
    """
    all_projects = projects.list_projects()
    history = training.get_history()

    completed = [run for run in history if run.get('status') == 'completed']
    accuracies = [
        run['metrics']['mAP50'] for run in completed
        if isinstance(run.get('metrics'), dict) and run['metrics'].get('mAP50') is not None
    ]

    active = []
    for project in all_projects:
        status = training.get_status(project['name'])
        if status and status.get('status') in training.ACTIVE_STATUSES:
            active.append({
                'project_name': project['name'],
                'model_name': status.get('model_name'),
                'current_epoch': status.get('current_epoch'),
                'total_epochs': status.get('total_epochs'),
            })

    return ok({
        'project_count': len(all_projects),
        'total_images': sum(p.get('total_images', 0) for p in all_projects),
        'annotated_images': sum(p.get('annotated_images', 0) for p in all_projects),
        'total_annotations': sum(p.get('total_annotations', 0) for p in all_projects),
        'training_runs': len(history),
        'completed_runs': len(completed),
        'failed_runs': len([r for r in history if r.get('status') == 'failed']),
        'active_runs': active,
        'average_map50': round(sum(accuracies) / len(accuracies), 4) if accuracies else None,
        'recent_runs': history[:10],
    })
