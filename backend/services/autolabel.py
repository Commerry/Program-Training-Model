"""
Model-assisted labelling.

Runs a model the project already trained over its own unlabelled images and
writes the predictions back as ordinary annotations, marked so a human can see
they were not drawn by hand.

This is the single largest time saver in the tool. Drawing 2000 boxes by hand
is days of work; correcting 2000 pre-drawn boxes is hours. The predictions are
deliberately written as normal editable regions rather than a separate
"suggestion" layer, so the annotator's existing workflow needs no new concepts.

The work happens in a background thread with a progress file, for the same
reason training does: a request that runs for minutes is a request that times
out.
"""

import threading
import time
from datetime import datetime
from pathlib import Path

from services import projects, training
from services.atomicio import read_json, write_json_best_effort
from services.projects import ProjectError

# One job per project at a time; the same lock discipline as training.
_locks = {}
_locks_guard = threading.Lock()


def _lock(name):
    with _locks_guard:
        return _locks.setdefault(name, threading.Lock())


def status_path(name):
    return projects.training_dir(name) / 'autolabel_status.json'


def get_status(name):
    projects.get_project(name)
    return read_json(status_path(name))


def _write_status(name, **fields):
    status = read_json(status_path(name)) or {}
    status.update(fields)
    write_json_best_effort(status_path(name), status)
    return status


def _run_for_weights(name, weights):
    """The history entry that produced these weights, if any."""
    for run in training.get_history(name):
        best = run.get('best_model')
        if best and Path(best) == Path(weights):
            return run
    return None


def _resolve_model(name, model_path=None):
    """
    The weights to label with: an explicit path, or the project's best run.

    Only a completed run is used. Labelling with a half-trained model
    produces confident nonsense that a human then has to undo, which is worse
    than starting from nothing.
    """
    if model_path:
        # Widened from this project to the whole projects tree. A new project
        # has no model of its own, which is exactly when pre-labelling would
        # save the most work, and a model trained on a similar job usually
        # gets most of the boxes close enough to be worth correcting. The
        # containment and extension checks are the same either way, so nothing
        # outside the tree becomes reachable.
        return training.resolve_trained_model(model_path)

    candidates = [run for run in training.get_history(name)
                  if run.get('status') == 'completed' and run.get('best_model')]
    if not candidates:
        available = [m for m in training.list_all_models()
                     if m['project'] != name and m['checkpoint'] == 'best']
        if available:
            suggestion = available[0]
            raise ProjectError(
                'This project has no trained model yet. Pick one from another '
                f'project to pre-label with -- "{suggestion["label"]}" from '
                f'{suggestion["project"]} is the most recent -- or train this '
                'project first.'
            )
        raise ProjectError(
            'No completed training run anywhere yet. Annotate some images by '
            'hand, train once, then use that model to pre-label the rest.'
        )

    # Best by mAP, falling back to most recent.
    candidates.sort(
        key=lambda r: ((r.get('metrics') or {}).get('mAP50') or -1,
                       r.get('completed_at') or ''),
        reverse=True,
    )
    best = candidates[0]
    path = Path(best['best_model'])
    if not path.is_file():
        raise ProjectError(
            f'The weights for run "{best["model_name"]}" are missing from disk.'
        )
    return path


def preview(name, model_path=None, score_threshold=0.4, img_size=None,
            sample=5, batch=None):
    """
    Run the model over a few images and report what it would draw. Write nothing.

    Asking "does auto-labelling work here" used to be unanswerable on a project
    that was already fully annotated: the pass only touches unlabelled images,
    so it refused with nothing to do, and the only way to see the model's
    output was to let it overwrite work drawn by hand. That is a bad trade for
    a question this cheap to answer.

    Prefers images that already have boxes, because those come with the answer:
    what a person drew is right there to compare against.
    """
    projects.get_project(name)
    weights = _resolve_model(name, model_path)

    try:
        score_threshold = min(max(float(score_threshold), 0.05), 0.95)
    except (TypeError, ValueError):
        score_threshold = 0.4

    run = _run_for_weights(name, weights)
    if not img_size:
        img_size = (run or {}).get('img_size') or 640
    img_size = max(64, min(int(img_size), 2048))

    entries = [e for e in projects.list_images(name) if not e['augmented']]
    if batch is not None:
        try:
            entries = [e for e in entries if e.get('batch') == int(batch)]
        except (TypeError, ValueError):
            pass
    if not entries:
        raise ProjectError('No photographs to try it on.')

    # Annotated first: those are the ones a prediction can be judged against.
    entries.sort(key=lambda e: (not e['annotated'], e['filename']))
    entries = entries[:max(1, min(int(sample), 20))]

    classes = _classes_for(name, weights)
    predict = _make_predictor(weights, img_size, score_threshold)

    results = []
    for entry in entries:
        path = projects.images_dir(name) / entry['filename']
        try:
            found = predict(path, classes)
        except Exception as exc:  # noqa: BLE001 - one bad image is not fatal
            results.append({'filename': entry['filename'], 'error': str(exc)})
            continue
        results.append({
            'filename': entry['filename'],
            'drawn_by_hand': entry['regions_count'],
            'model_found': len(found),
            'best_score': round(max((r['score'] for r in found), default=0.0), 4),
            'tags': sorted({r['tag'] for r in found}),
        })

    with_something = [r for r in results if r.get('model_found')]
    if not with_something:
        verdict = (f'Found nothing on any of {len(results)} image(s) above '
                   f'{score_threshold:.2f}. Either this model has not learned '
                   'these objects, or the threshold is too high.')
    else:
        verdict = (f'Found objects on {len(with_something)} of {len(results)} '
                   f'image(s). Compare the counts below against what was drawn '
                   'by hand before letting it label the rest.')

    return {
        'model': str(weights),
        'model_name': Path(weights).name,
        'score_threshold': score_threshold,
        'img_size': img_size,
        'results': results,
        'verdict': verdict,
        'usable': bool(with_something),
    }


def start(name, model_path=None, score_threshold=0.4, overwrite=False,
          only_unannotated=True, img_size=None, limit=None, batch=None):
    """
    Kick off a background labelling pass. Returns the initial status.

    `batch` narrows the pass to one upload. Once a project is being extended
    rather than built, "everything unlabelled" and "the images I just added"
    stop being the same set: a few pictures skipped months ago are still
    unlabelled, and running a model over them alongside today's import mixes
    two decisions into one review.
    """
    projects.get_project(name)

    try:
        score_threshold = min(max(float(score_threshold), 0.05), 0.95)
    except (TypeError, ValueError):
        score_threshold = 0.4

    weights = _resolve_model(name, model_path)

    # Infer at the size the model was trained at unless told otherwise. A
    # detector is sensitive to object scale, and running a model trained at
    # 320 over images letterboxed to 640 presents every object at roughly
    # twice the size it ever saw — which measurably suppresses detections.
    run = _run_for_weights(name, weights)
    if not img_size:
        img_size = (run or {}).get('img_size') or 640
    img_size = max(64, min(int(img_size), 2048))

    current = get_status(name)
    if current and current.get('status') == 'running':
        raise ProjectError('Auto-labelling is already running for this project')

    if batch is not None:
        try:
            batch = int(batch)
        except (TypeError, ValueError):
            batch = None

    entries = projects.list_images(name)
    if batch is not None:
        entries = [e for e in entries if e.get('batch') == batch]

    targets = [
        entry['filename'] for entry in entries
        if (not entry['annotated'] or (overwrite and not only_unannotated))
        and not entry['augmented']
    ]
    if only_unannotated:
        targets = [f for f in targets
                   if not projects.read_annotation(name, f).get('annotated')]
    if limit:
        targets = targets[:int(limit)]

    if not targets:
        raise ProjectError(
            'Every image already has annotations. Enable "overwrite" if you '
            'want to replace them.'
        )

    lock = _lock(name)
    if not lock.acquire(blocking=False):
        raise ProjectError('Auto-labelling is already starting for this project')

    status = {
        'status': 'running',
        'model': str(weights),
        'model_name': weights.parent.parent.name,
        'score_threshold': score_threshold,
        'img_size': img_size,
        'total': len(targets),
        'processed': 0,
        'labelled': 0,
        'boxes': 0,
        'skipped': 0,
        'started_at': datetime.now().isoformat(),
        'completed_at': None,
        'error': None,
    }
    write_json_best_effort(status_path(name), status)

    thread = threading.Thread(
        target=_run,
        args=(name, weights, targets, score_threshold, img_size, lock),
        name=f'autolabel-{name}',
        daemon=True,
    )
    thread.start()
    return status


def cancel(name):
    projects.get_project(name)
    status = read_json(status_path(name))
    if not status or status.get('status') != 'running':
        raise ProjectError('Auto-labelling is not running')
    _write_status(name, status='cancelling')
    return {'message': 'Cancelling'}


def _run(name, weights, targets, score_threshold, img_size, lock):
    """Worker body. Never raises — failures are reported through the status."""
    try:
        classes = _classes_for(name, weights)
        predict = _make_predictor(weights, img_size, score_threshold)

        images_dir = projects.images_dir(name)
        labelled = boxes_written = skipped = 0
        # Kept so a pass that silently found nothing can be diagnosed; a bare
        # "skipped: 10" tells the user nothing about why.
        errors = []

        for index, filename in enumerate(targets, start=1):
            current = read_json(status_path(name)) or {}
            if current.get('status') == 'cancelling':
                _write_status(name, status='cancelled',
                              completed_at=datetime.now().isoformat())
                return

            image_path = images_dir / filename
            try:
                detections = predict(image_path, classes)
            except Exception as exc:  # noqa: BLE001 - one bad image must not end the pass
                if len(errors) < 5:
                    errors.append(f'{filename}: {type(exc).__name__}: {exc}')
                skipped += 1
                detections = []

            if detections:
                existing = projects.read_annotation(name, filename)
                projects.write_annotation(name, filename, {
                    **existing,
                    'filename': filename,
                    'regions': detections,
                    'annotated': True,
                    # Kept so the UI can mark these for review and so a later
                    # pass can tell its own output from a human's.
                    'auto_labelled': True,
                    'auto_label': {
                        'model': str(weights),
                        'score_threshold': score_threshold,
                        'at': datetime.now().isoformat(),
                    },
                    'updated_at': datetime.now().isoformat(),
                })
                projects._update_index_entry(  # noqa: SLF001 - same package
                    name, filename, projects.read_annotation(name, filename))
                labelled += 1
                boxes_written += len(detections)
            else:
                skipped += 1

            if index % 5 == 0 or index == len(targets):
                _write_status(name, processed=index, labelled=labelled,
                              boxes=boxes_written, skipped=skipped, errors=errors)

        projects.rebuild_index(name)
        projects.refresh_stats(name)

        # A run that labels nothing has always looked identical to one that
        # was never asked to do anything: status completed, no message, and a
        # count of zero with no hint whether the model found nothing, the
        # threshold was too high, or the wrong model was chosen.
        if not targets:
            message = 'Nothing to do: every image already has annotations.'
        elif labelled == 0:
            message = (
                f'Looked at {len(targets)} image(s) and found nothing above '
                f'{score_threshold:.2f}. Either the model has not learned '
                'these objects, or the threshold is too high -- try lowering '
                'it, or pick a different model.'
            )
        else:
            message = (f'Labelled {labelled} of {len(targets)} image(s) with '
                       f'{boxes_written} box(es). Check them before training.')

        _write_status(name, status='completed', processed=len(targets),
                      labelled=labelled, boxes=boxes_written, skipped=skipped,
                      errors=errors, message=message,
                      completed_at=datetime.now().isoformat())
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        _write_status(name, status='failed', error=str(exc),
                      completed_at=datetime.now().isoformat())
    finally:
        try:
            lock.release()
        except RuntimeError:
            pass


def _classes_for(name, weights):
    """Class names for the model, preferring what the run recorded."""
    for run in training.get_history(name):
        if run.get('best_model') and Path(run['best_model']) == weights:
            if run.get('classes'):
                return [str(c) for c in run['classes']]
    return projects.class_names(name)


def _make_predictor(weights, img_size, score_threshold):
    """
    Return f(image_path, classes) -> [region dicts in image pixels].

    Torch and ultralytics are imported here rather than at module scope so the
    web process does not load them until someone actually asks for labelling.
    """
    suffix = weights.suffix.lower()

    if suffix == '.pth':
        import torch

        from services.imaging import imread
        from training.frcnn_lib import create_model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(str(weights), map_location=device, weights_only=True)
        class_weight = state.get('roi_heads.box_predictor.cls_score.weight')
        if class_weight is None:
            raise RuntimeError('Not a Faster R-CNN checkpoint')
        num_classes = int(class_weight.shape[0]) - 1
        model = create_model(num_classes=num_classes, pretrained=False, img_size=img_size)
        model.load_state_dict(state)
        model.to(device).eval()

        def predict(path, classes):
            import cv2
            bgr = imread(path)
            if bgr is None:
                return []
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(device)
            with torch.no_grad():
                out = model([tensor])[0]
            regions = []
            for box, score, label in zip(out['boxes'].cpu().numpy(),
                                         out['scores'].cpu().numpy(),
                                         out['labels'].cpu().numpy()):
                if float(score) < score_threshold:
                    continue
                index = int(label) - 1  # 0 is background
                if not 0 <= index < len(classes):
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                regions.append({'tag': str(classes[index]), 'x': x1, 'y': y1,
                                'width': x2 - x1, 'height': y2 - y1,
                                'score': round(float(score), 4)})
            return regions

        return predict

    from ultralytics import YOLO
    model = YOLO(str(weights))
    embedded = getattr(model, 'names', None)

    def predict(path, classes):
        names = classes
        if isinstance(embedded, dict) and embedded:
            names = [str(embedded[k]) for k in sorted(embedded)]
        result = model.predict(str(path), conf=score_threshold, imgsz=img_size,
                               verbose=False)[0]
        regions = []
        if result.boxes is None or not len(result.boxes):
            return regions
        for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy(),
                                   result.boxes.cls.cpu().numpy().astype(int)):
            if not 0 <= int(cls) < len(names):
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            regions.append({'tag': str(names[int(cls)]), 'x': x1, 'y': y1,
                            'width': x2 - x1, 'height': y2 - y1,
                            'score': round(float(score), 4)})
        return regions

    return predict
