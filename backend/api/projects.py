"""Project, image, annotation and dataset endpoints."""

import tempfile
from pathlib import Path

from flask import Blueprint, request, send_file

from api import login_required_api, ok
from services import augment, autolabel, dataset, projects
from services.projects import ProjectError

projects_bp = Blueprint('projects', __name__, url_prefix='/projects')
projects_bp.before_request(login_required_api(lambda: None))


# ── projects ────────────────────────────────────────────────────────────────

@projects_bp.get('')
def list_projects():
    return ok({'projects': projects.list_projects()})


@projects_bp.post('')
def create_project():
    data = request.get_json(silent=True) or {}
    project = projects.create_project(data.get('name'), data.get('description', ''))
    return ok({'message': f'Project {project["name"]} created', 'project': project})


@projects_bp.get('/<project_name>')
def get_project(project_name):
    return ok({'project': projects.get_project(project_name)})


@projects_bp.delete('/<project_name>')
def delete_project(project_name):
    return ok(projects.delete_project(project_name))


@projects_bp.get('/<project_name>/tags')
def get_tags(project_name):
    project = projects.get_project(project_name)
    tags = project.get('tags', {})
    return ok({'tags': sorted(tags), 'tags_detail': tags})


@projects_bp.get('/<project_name>/dataset-summary')
def dataset_summary(project_name):
    return ok(projects.dataset_summary(project_name))


@projects_bp.post('/<project_name>/rescan')
def rescan_project(project_name):
    """Re-read this project from disk; optionally clear counts that cannot be."""
    data = request.get_json(silent=True) or {}
    return ok(projects.rescan(
        project_name,
        clear_if_missing=bool(data.get('clear_if_missing')),
    ))


# ── images ──────────────────────────────────────────────────────────────────

@projects_bp.get('/<project_name>/images')
def list_images(project_name):
    return ok({'images': projects.list_images(project_name)})


@projects_bp.post('/<project_name>/images')
def import_images(project_name):
    files = request.files.getlist('images')
    if not files or all(not f.filename for f in files):
        raise ProjectError('No images were uploaded')
    result = projects.import_images(project_name, files)
    return ok({'message': f'Imported {result["imported_count"]} images', **result})


# The default converter is used rather than <path:...>: generated file names
# never contain a slash, and a greedy converter would let
# /images/a.jpg/annotations match the plain image route.
@projects_bp.get('/<project_name>/images/<filename>/raw')
def serve_image(project_name, filename):
    projects.get_project(project_name)
    path = projects.images_dir(project_name) / projects.safe_filename(filename)
    if not path.is_file():
        raise ProjectError('Image not found', status=404)
    # max_age lets the browser reuse images while scrolling a large gallery;
    # file names are unique per upload so a stale cache is not possible.
    return send_file(str(path), max_age=86400)


@projects_bp.get('/<project_name>/images/<filename>')
def get_image_data(project_name, filename):
    return ok({'data': projects.get_image_data(project_name, filename)})


@projects_bp.delete('/<project_name>/images/<filename>')
def delete_image(project_name, filename):
    return ok(projects.delete_image(project_name, filename))


@projects_bp.post('/<project_name>/images/<filename>/annotations')
def save_annotations(project_name, filename):
    data = request.get_json(silent=True) or {}
    result = projects.save_annotations(project_name, filename, data.get('regions', []))
    return ok({'message': 'Annotations saved', **result})


# ── augmentation ────────────────────────────────────────────────────────────

@projects_bp.get('/<project_name>/augment-color/tones')
def list_tones(project_name):
    projects.get_project(project_name)
    return ok({'tones': augment.ALL_COLOR_TONES})


@projects_bp.post('/<project_name>/augment-color')
def augment_color(project_name):
    data = request.get_json(silent=True) or {}
    result = augment.augment_color_images(
        project_name,
        source_filenames=data.get('source_filenames'),
        tones=data.get('tones'),
        variants_per_tone=data.get('variants_per_tone', 3),
        strength=data.get('strength', 1.0),
        require_all_annotated=bool(data.get('require_all_annotated', True)),
    )
    return ok(result)


# ── dataset zip ─────────────────────────────────────────────────────────────

@projects_bp.post('/<project_name>/export')
def export_dataset(project_name):
    result = dataset.export_dataset(project_name)
    return send_file(result['export_file'], as_attachment=True,
                     download_name=result['export_name'])


@projects_bp.post('/<project_name>/import-dataset')
def import_dataset(project_name):
    file = request.files.get('file')
    if not file or not file.filename:
        raise ProjectError('No file was uploaded')
    if Path(file.filename).suffix.lower() != '.zip':
        raise ProjectError('Dataset imports must be a .zip produced by Export')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        temp_path = Path(tmp.name)
    try:
        file.save(str(temp_path))
        return ok(dataset.import_dataset(project_name, temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


# ── auto-labelling ──────────────────────────────────────────────────────────

@projects_bp.post('/<project_name>/auto-label')
def start_auto_label(project_name):
    """Pre-label unannotated images with a model this project already trained."""
    data = request.get_json(silent=True) or {}
    return ok({'job': autolabel.start(
        project_name,
        model_path=data.get('model_path'),
        score_threshold=data.get('score_threshold', 0.4),
        only_unannotated=bool(data.get('only_unannotated', True)),
        overwrite=bool(data.get('overwrite', False)),
        # Omitted by default so the service can use the size the model was
        # actually trained at.
        img_size=data.get('img_size'),
        limit=data.get('limit'),
    )})


@projects_bp.get('/<project_name>/auto-label')
def auto_label_status(project_name):
    status = autolabel.get_status(project_name)
    return ok({'job': status, 'running': bool(status and status.get('status') == 'running')})


@projects_bp.post('/<project_name>/auto-label/cancel')
def cancel_auto_label(project_name):
    return ok(autolabel.cancel(project_name))


@projects_bp.post('/<project_name>/prepare-dataset')
def prepare_dataset(project_name):
    """Build the train/val split without starting a run, to preview the result."""
    return ok({'dataset': dataset.build_yolo_dataset(project_name)})
