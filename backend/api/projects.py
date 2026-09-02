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


@projects_bp.post('/<project_name>/images/<filename>/detect')
def detect_on_image(project_name, filename):
    """
    Run a model over one image already in the project, and hand back regions.

    Deliberately separate from both auto-labelling and from saving. Auto-label
    is a bulk background pass over everything unannotated; this is one image,
    on demand, while somebody is looking at it -- so it can be tried on a
    single picture to see whether a model is worth trusting before it is turned
    loose on a thousand.

    Nothing is written. The boxes come back in the same shape the annotation
    editor uses, and it is the person looking at them who decides.
    """
    from services import inference, training
    from services.imaging import imread

    data = request.get_json(silent=True) or {}
    model_path = training.resolve_trained_model(data.get('model_path'))

    path = projects.images_dir(project_name) / projects.safe_filename(filename)
    if not path.is_file():
        raise ProjectError('No such image in this project', status=404)
    image = imread(path)
    if image is None:
        raise ProjectError('That image could not be read')

    detections, labels = inference.detect_frame(
        model_path, image,
        score_threshold=data.get('score_threshold', 0.4),
        label_names=data.get('label_names'),
        img_size=data.get('img_size'))

    # The editor works in x/y/width/height with a tag; the detector answers in
    # corners with a class name. Converting here keeps that difference out of
    # the page.
    regions = []
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        regions.append({
            'tag': detection['label_name'],
            'x': int(x1), 'y': int(y1),
            'width': int(x2 - x1), 'height': int(y2 - y1),
            'score': detection['score'],
        })

    return ok({
        'regions': regions,
        'count': len(regions),
        'reading': inference.reading_of(detections),
        'label_names': labels,
        'model': Path(model_path).name,
        'width': int(image.shape[1]),
        'height': int(image.shape[0]),
    })


@projects_bp.post('/<project_name>/auto-label/preview')
def preview_auto_label(project_name):
    """
    Try the model on a few images and report what it would draw. Writes nothing.

    The pass itself only touches unlabelled images, so on a project that is
    already fully annotated there was no way to find out whether a model was
    worth using short of letting it overwrite work drawn by hand.
    """
    data = request.get_json(silent=True) or {}
    return ok(autolabel.preview(
        project_name,
        model_path=data.get('model_path'),
        score_threshold=data.get('score_threshold', 0.4),
        sample=data.get('sample', 5),
        batch=data.get('batch'),
    ))


@projects_bp.post('/<project_name>/dataset-import/preview')
def preview_dataset_import(project_name):
    """
    What is in an export folder, without copying anything.

    Answering "is this the right folder, and will it come in whole" before
    moving six thousand files is worth the pass it costs.
    """
    from services import datasetimport
    data = request.get_json(silent=True) or {}
    projects.get_project(project_name)
    return ok(datasetimport.preview(data.get('folder')))


@projects_bp.post('/<project_name>/dataset-import')
def start_dataset_import(project_name):
    """Copy an annotated dataset in, in the background."""
    from services import datasetimport
    data = request.get_json(silent=True) or {}
    return ok({'job': datasetimport.start(project_name, data.get('folder'),
                                          limit=data.get('limit'))})


@projects_bp.get('/<project_name>/dataset-import')
def dataset_import_status(project_name):
    from services import datasetimport
    status = datasetimport.get_status(project_name)
    return ok({'job': status,
               'running': bool(status and status.get('status') == 'running')})


@projects_bp.post('/<project_name>/dataset-import/cancel')
def cancel_dataset_import(project_name):
    from services import datasetimport
    return ok(datasetimport.cancel(project_name))


@projects_bp.get('/<project_name>/review/queue')
def review_queue(project_name):
    """
    Pre-labelled pictures nobody has checked yet, the least certain first.

    A picture the model was confident about teaches nothing whether it was
    right or wrong; the ones it hesitated over are where the next bit of
    accuracy is.
    """
    from services import review
    return ok(review.queue(project_name,
                           limit=request.args.get('limit', 50, type=int)))


@projects_bp.get('/<project_name>/review/summary')
def review_summary(project_name):
    """
    What people have corrected, per class.

    Arithmetic over what was actually changed, with no contribution from the
    model: a class mostly relabelled is being confused, one mostly added is
    being missed, one mostly deleted is being imagined. Three problems, three
    fixes, and one accuracy number tells them apart from none of the others.
    """
    from services import review
    return ok(review.project_summary(project_name))


@projects_bp.get('/<project_name>/class-accuracy')
def class_accuracy(project_name):
    """How well the newest finished run did on each class."""
    from services import training
    return ok(training.class_accuracy(project_name))


@projects_bp.post('/<project_name>/images/delete')
def delete_images(project_name):
    """
    Delete a list of images, or every generated copy.

    A POST rather than a DELETE because the list of names goes in the body: a
    few hundred filenames do not belong in a URL.
    """
    data = request.get_json(silent=True) or {}
    return ok(projects.delete_images(
        project_name,
        filenames=data.get('filenames'),
        only_generated=bool(data.get('only_generated')),
    ))


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
    """
    A zipped dataset, whether this application exported it or not.

    It used to take only its own export and refuse everything else, which is
    the wrong way round: the datasets worth importing are the ones from
    somewhere else. A zip that carries this application's dataset.json goes
    the direct route; anything else is unpacked and read as YOLO, COCO or
    Pascal VOC, the same as a folder would be.
    """
    from services import datasetimport

    file = request.files.get('file')
    if not file or not file.filename:
        raise ProjectError('No file was uploaded')
    if Path(file.filename).suffix.lower() != '.zip':
        raise ProjectError(
            f'"{file.filename}" is not a .zip. Zip the export folder and '
            'upload that, or give its path on this machine instead — a '
            'folder of six thousand pictures does not need zipping first.')

    projects.get_project(project_name)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        temp_path = Path(tmp.name)
    try:
        file.save(str(temp_path))
        if datasetimport.is_own_export(temp_path):
            return ok(dataset.import_dataset(project_name, temp_path))
        # Unpacked to a staging folder the background reader can work from:
        # thousands of images take minutes, and the request would time out.
        folder = datasetimport.unpack(temp_path, project_name)
        return ok({'job': datasetimport.start(project_name, folder)})
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
        batch=data.get('batch'),
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
