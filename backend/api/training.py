"""Training control, results and history endpoints."""

from flask import Blueprint, request, send_file

from api import login_required_api, ok
from services import trainaug, training
from services.projects import ProjectError

training_bp = Blueprint('training', __name__, url_prefix='/projects/<project_name>/training')
training_bp.before_request(login_required_api(lambda: None))


@training_bp.get('/options')
def options(project_name):
    """Model types and export formats the backend actually supports."""
    # The augmentation advice comes with the options because it is decided
    # from this project's class names, not from a global setting: whether
    # mirroring is safe depends on whether the classes read as text.
    return ok({
        'model_types': training.MODEL_TYPES,
        'yolo_export_formats': sorted(training.YOLO_EXPORT_FORMATS),
        'frcnn_export_formats': sorted(training.FRCNN_EXPORT_FORMATS),
        'augmentation': trainaug.recommend(project_name),
    })


@training_bp.post('/start')
def start(project_name):
    data = request.get_json(silent=True) or {}
    return ok(training.start(
        project_name,
        model_type=data.get('model_type', 'yolo11s'),
        epochs=data.get('epochs', 100),
        batch_size=data.get('batch_size', 16),
        img_size=data.get('img_size', 640),
        learning_rate=data.get('learning_rate', 0.01),
        export_formats=data.get('export_formats') or ['pt'],
        model_name=data.get('model_name', ''),
        augmentation=data.get('augmentation'),
        generate_filters=data.get('generate_filters'),
    ))


@training_bp.get('/status')
def status(project_name):
    current = training.get_status(project_name)
    if not current:
        # Not an error: the project simply has not been trained yet.
        return ok({'status': None, 'has_run': False})
    return ok({'status': current, 'has_run': True})


@training_bp.post('/stop')
def stop(project_name):
    return ok(training.stop(project_name))


@training_bp.post('/reset')
def reset(project_name):
    return ok(training.reset_status(project_name))


@training_bp.get('/logs')
def logs(project_name):
    try:
        last_n = int(request.args.get('last_n', 200))
    except (TypeError, ValueError):
        last_n = 200
    return ok({'logs': training.get_logs(project_name, min(max(last_n, 1), 5000))})


@training_bp.get('/models')
def models(project_name):
    return ok({'models': training.list_models(project_name)})


@training_bp.get('/models/download')
def download_model(project_name):
    path = training.resolve_model_path(project_name, request.args.get('path'))
    return send_file(str(path), as_attachment=True, download_name=path.name)


@training_bp.get('/history')
def project_history(project_name):
    return ok({'history': training.get_history(project_name)})
