"""Model testing and cross-project reporting endpoints."""

from flask import Blueprint, request

from api import login_required_api, ok
from services import inference, projects, training
from services.projects import ProjectError

inference_bp = Blueprint('inference', __name__)
inference_bp.before_request(login_required_api(lambda: None))


@inference_bp.post('/models/test')
def test_model():
    labels_text = (request.form.get('label_names') or '').strip()
    label_names = [part.strip() for part in labels_text.replace('\n', ',').split(',')
                   if part.strip()]

    # Parsed defensively: a non-numeric value used to raise straight out of
    # int() as a 500 rather than a 400.
    try:
        img_size = int(request.form.get('img_size') or 640)
    except (TypeError, ValueError):
        raise ProjectError('img_size must be a number')
    if not 64 <= img_size <= 4096:
        raise ProjectError('img_size must be between 64 and 4096')

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
    ))


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
