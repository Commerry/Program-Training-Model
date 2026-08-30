"""
Project, image and annotation storage.

On-disk layout (one directory per project under PROJECTS_ROOT):

    <project>/
        project.json                 name, description, cached stats
        images/<file>                source images, never modified
        annotations/<file>.json      one file per image, boxes in image pixels
        training/                    configs, logs, weights (see training.py)
        exports/                     dataset zips

project.json holds cached counters only. The annotation files are the source of
truth and the counters are recomputed from them by refresh_stats().
"""

import math
import os
import random
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from config import IMAGE_EXTENSIONS, PROJECTS_ROOT
from services.atomicio import (
    AtomicWriteError, read_json, write_json, write_json_best_effort,
)
from services.imaging import image_size, imread

_SAFE_NAME = re.compile(r'^[A-Za-z0-9ก-๙][A-Za-z0-9ก-๙ _.\-]{0,63}$')

# Windows refuses to create a directory with any of these names, with or
# without an extension, and silently strips trailing dots and spaces from the
# rest. Both were reachable: "NUL" raised WinError 3 out of create_project as
# an opaque 500, and "proj." created a directory called "proj" that then could
# not be addressed by the name the caller was given back.
_WINDOWS_RESERVED = {
    'CON', 'PRN', 'AUX', 'NUL',
    *(f'COM{i}' for i in range(1, 10)),
    *(f'LPT{i}' for i in range(1, 10)),
}

# Characters Windows rejects in a path component. Checked separately from the
# name regex because file names (not just project names) go through it too.
_ILLEGAL_PATH_CHARS = set('<>:"/\\|?*') | {chr(c) for c in range(32)}


class ProjectError(Exception):
    """Raised for conditions the API should report as a 4xx, not a crash."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


# ── naming and paths ────────────────────────────────────────────────────────

def validate_name(name):
    """
    Reject anything that could escape PROJECTS_ROOT or break on Windows.

    Project names come straight from the URL, so this is the only thing
    standing between a request and an arbitrary filesystem path.
    """
    name = (name or '').strip()
    if not _SAFE_NAME.match(name):
        raise ProjectError(
            'Project name must be 1-64 characters and may contain letters, '
            'digits, spaces, dots, underscores and hyphens only.'
        )
    if name != Path(name).name or name in ('.', '..'):
        raise ProjectError('Invalid project name.')
    if name != name.rstrip('. '):
        raise ProjectError(
            'Project name cannot end with a dot or a space — Windows silently '
            'removes them, which would make the project unreachable.'
        )
    if name.split('.')[0].upper() in _WINDOWS_RESERVED:
        raise ProjectError(
            f'"{name}" is a reserved device name on Windows. Choose another name.'
        )
    return name


def project_dir(name):
    return PROJECTS_ROOT / validate_name(name)


def images_dir(name):
    return project_dir(name) / 'images'


def annotations_dir(name):
    return project_dir(name) / 'annotations'


def training_dir(name):
    return project_dir(name) / 'training'


def exports_dir(name):
    return project_dir(name) / 'exports'


def safe_filename(filename, max_length=120):
    """
    Allow only a bare file name that Windows will actually accept.

    Path.name alone is not enough: it happily returns "a*b", which passes here
    and then fails deep inside a worker after the dataset has been rebuilt and
    a process spawned. Rejecting it up front turns that into a clear 400.
    """
    filename = (filename or '').strip()
    if not filename or filename != Path(filename).name or filename in ('.', '..'):
        raise ProjectError('Invalid file name.')
    if len(filename) > max_length:
        raise ProjectError(f'Name is too long (limit {max_length} characters).')
    illegal = sorted(_ILLEGAL_PATH_CHARS & set(filename))
    if illegal:
        printable = ' '.join(repr(c) for c in illegal if c.isprintable())
        raise ProjectError(
            f'Name contains characters that are not allowed in a file name'
            + (f': {printable}' if printable else '.')
        )
    if filename != filename.rstrip('. '):
        raise ProjectError('Name cannot end with a dot or a space.')
    if Path(filename).stem.upper() in _WINDOWS_RESERVED:
        raise ProjectError(f'"{filename}" is a reserved device name on Windows.')
    return filename


def exists(name):
    return (project_dir(name) / 'project.json').exists()


def _require(name):
    if not exists(name):
        raise ProjectError(f'Project "{name}" not found', status=404)
    return validate_name(name)


# ── project.json ────────────────────────────────────────────────────────────

def _read_meta(name):
    meta = read_json(project_dir(name) / 'project.json', default={}) or {}
    meta['name'] = name
    # The stored path used to be a CWD-relative string, which broke as soon as
    # the server was started from a different directory. Always derive it.
    meta['path'] = str(project_dir(name))
    meta.setdefault('description', '')
    meta.setdefault('created_at', None)
    meta.setdefault('total_images', 0)
    meta.setdefault('total_annotations', 0)
    meta.setdefault('annotated_images', 0)
    meta.setdefault('tags', {})
    # The counters are a cache; this says whether the files behind them are
    # currently reachable. Without it a project whose folder has moved shows
    # "2232 images" beside an empty gallery and nothing explains the gap.
    meta['images_available'] = images_dir(name).is_dir()
    return meta


def _write_meta(name, meta):
    meta = dict(meta)
    meta['name'] = name
    meta['path'] = str(project_dir(name))
    meta['updated_at'] = datetime.now().isoformat()
    # Recomputable from the annotation files, so a failed write here is not
    # worth failing the request over.
    write_json_best_effort(project_dir(name) / 'project.json', meta)
    return meta


def list_projects():
    """
    Every directory under PROJECTS_ROOT that contains a project.json.

    Scanning the directory removes the need for the old projects_config.json
    index, which drifted out of sync whenever a project folder was added,
    renamed or deleted outside the app.
    """
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    projects = []
    for entry in sorted(PROJECTS_ROOT.iterdir()):
        if not entry.is_dir() or not (entry / 'project.json').exists():
            continue
        try:
            projects.append(_read_meta(entry.name))
        except ProjectError:
            continue  # directory name is not a legal project name — skip it
    return projects


def get_project(name):
    _require(name)
    return _read_meta(name)


def create_project(name, description=''):
    name = validate_name(name)
    if exists(name):
        raise ProjectError('Project name already exists')

    root = project_dir(name)
    (root / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'annotations').mkdir(parents=True, exist_ok=True)
    (root / 'training').mkdir(parents=True, exist_ok=True)

    return _write_meta(name, {
        'description': description or '',
        'created_at': datetime.now().isoformat(),
        'total_images': 0,
        'total_annotations': 0,
        'annotated_images': 0,
        'tags': {},
    })


# Windows refuses to delete or rename a file another handle has open, and this
# app hands image files to Flask's send_file, which holds them until the
# response is fully written. A gallery request that is still finishing is
# enough to make an immediate delete fail, so deletes retry briefly.
_DELETE_RETRIES = (0.05, 0.15, 0.3, 0.6, 1.0)


def _remove_with_retry(remove, target):
    last = None
    for delay in (0.0,) + _DELETE_RETRIES:
        if delay:
            time.sleep(delay * (0.5 + random.random()))
        try:
            remove()
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            last = exc
    raise ProjectError(
        f'Could not delete {target}: a file is still open, most likely because '
        'an image is being served right now. Try again in a moment.',
        status=409,
    ) from last


def delete_project(name):
    _require(name)
    directory = project_dir(name)
    _remove_with_retry(lambda: shutil.rmtree(directory), f'project "{name}"')
    return {'message': f'Project {name} deleted'}


# ── gallery index ───────────────────────────────────────────────────────────
#
# Rendering the gallery once meant opening every annotation file: 2232 opens
# for the reference dataset, which measured 18.8 seconds cold on Windows
# (roughly 8 ms per open, dominated by real-time antivirus scanning) and was
# still 300 ms warm. The per-image summary the gallery needs is tiny, so it is
# kept in one index file and updated in place as annotations change.
#
# The index is a cache, never the source of truth. It carries a fingerprint of
# how many image and annotation files exist; if that no longer matches the
# directory, the index is rebuilt from the annotation files. Files edited
# outside the app without changing those counts are picked up by
# rebuild_index(), which the training and dataset paths always call.

INDEX_VERSION = 4

# A thumbnail a few hundred pixels wide cannot usefully show more outlines than
# this, and an image with hundreds of boxes would otherwise dominate the index.
MAX_INDEXED_BOXES = 60


def index_path(name):
    return project_dir(name) / 'index.json'


def _scan_stats(directory, suffixes=None):
    """
    (count, newest mtime) for a directory, in one pass.

    DirEntry caches the stat data the directory walk already returned, so this
    costs a single scan rather than a stat syscall per file.
    """
    count = 0
    newest = 0.0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                if suffixes is not None and os.path.splitext(entry.name)[1].lower() not in suffixes:
                    continue
                count += 1
                try:
                    newest = max(newest, entry.stat().st_mtime)
                except OSError:
                    pass
    except OSError:
        return 0, 0.0
    return count, newest


def _fingerprint(name):
    """
    What the index was built from.

    Counts catch additions and deletions; the newest annotation mtime catches
    a file rewritten in place. Every incremental index update recomputes and
    stores this, so ordinary edits through the app do not force a rebuild.
    """
    image_count, _ = _scan_stats(images_dir(name), IMAGE_EXTENSIONS)
    ann_count, ann_newest = _scan_stats(annotations_dir(name))
    return {
        'images': image_count,
        'annotations': ann_count,
        # Rounded: some filesystems store a coarser timestamp than they report.
        'annotations_mtime': round(ann_newest, 3),
    }


def _entry_from_annotation(filename, ann, size_kb=None):
    """
    The gallery- and stats-relevant summary of one annotation file.

    tag_counts (boxes per class in this image) is stored rather than a bare
    tag list so refresh_stats() is pure arithmetic over the index and never
    has to reopen an annotation file.
    """
    regions = ann.get('regions', []) or []
    tag_counts = {}
    boxes = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        tag = str(region.get('tag') or '').strip()
        if tag:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        try:
            # Stored as whole pixels: the gallery draws these a couple of
            # hundred pixels wide, so sub-pixel precision is invisible and
            # rounding roughly halves the size of the index.
            boxes.append([
                round(float(region['x'])),
                round(float(region['y'])),
                round(float(region['width'])),
                round(float(region['height'])),
                tag,
            ])
        except (KeyError, TypeError, ValueError):
            continue

    return {
        'annotated': bool(ann.get('annotated') and regions),
        'regions_count': len(regions),
        'tag_counts': tag_counts,
        # Kept in the index so the gallery can draw every thumbnail's boxes
        # from the one request it already makes, instead of one per image.
        'boxes': boxes[:MAX_INDEXED_BOXES],
        'width': ann.get('width'),
        'height': ann.get('height'),
        'augmented': bool(ann.get('augmented')),
        # Which upload this image arrived in. Images from before batches were
        # recorded have none, which reads as "the original set" and is exactly
        # what they are.
        'batch': ann.get('batch'),
        'imported_at': ann.get('imported_at'),
        'size_kb': size_kb,
    }


def rebuild_index(name):
    """
    Read every annotation file and write a fresh index.

    If the images directory is not there at all, the existing index is left
    alone. A missing directory means the data is unreachable — an unmounted
    drive, a mistyped PROJECTS_ROOT, a checkout without the image files — and
    writing an empty index would cache that emptiness as if it were the truth.
    """
    if not images_dir(name).is_dir():
        return read_json(index_path(name)) or {'version': INDEX_VERSION,
                                               'fingerprint': None, 'entries': {}}

    entries = {}
    for img_file in iter_image_files(name):
        ann = read_annotation(name, img_file.name)
        try:
            size_kb = round(img_file.stat().st_size / 1024, 1)
        except OSError:
            size_kb = None
        entry = _entry_from_annotation(img_file.name, ann, size_kb)

        # Dimensions are required by the dataset builder. Filling them in here
        # means the expensive header read happens once per image, ever.
        if not (entry['width'] and entry['height']):
            size = image_size(img_file)
            if size:
                entry['width'], entry['height'] = size
                ann['width'], ann['height'] = size
                write_annotation(name, img_file.name, ann)
        entries[img_file.name] = entry

    index = {
        'version': INDEX_VERSION,
        'fingerprint': _fingerprint(name),
        'entries': entries,
    }
    write_json_best_effort(index_path(name), index, indent=None)
    return index


def load_index(name, allow_rebuild=True):
    """The index, rebuilt if it is missing, stale or from an older version."""
    index = read_json(index_path(name))
    fresh = (
        isinstance(index, dict)
        and index.get('version') == INDEX_VERSION
        and isinstance(index.get('entries'), dict)
        and index.get('fingerprint') == _fingerprint(name)
    )
    if fresh:
        return index
    if not allow_rebuild:
        return None
    return rebuild_index(name)


def _update_index_entry(name, filename, ann):
    """
    Patch one entry after a write, leaving the rest of the index alone.

    Falls back to a full rebuild only when the index is missing or stale, so
    the common case stays a single read and a single write.
    """
    index = read_json(index_path(name))
    if not (isinstance(index, dict) and index.get('version') == INDEX_VERSION
            and isinstance(index.get('entries'), dict)):
        return rebuild_index(name)

    try:
        size_kb = round((images_dir(name) / filename).stat().st_size / 1024, 1)
    except OSError:
        size_kb = None

    index['entries'][filename] = _entry_from_annotation(filename, ann, size_kb)
    index['fingerprint'] = _fingerprint(name)
    write_json_best_effort(index_path(name), index, indent=None)
    return index


def _remove_index_entry(name, filename):
    index = read_json(index_path(name))
    if not (isinstance(index, dict) and isinstance(index.get('entries'), dict)):
        return
    index['entries'].pop(filename, None)
    index['fingerprint'] = _fingerprint(name)
    write_json_best_effort(index_path(name), index, indent=None)


# ── annotations ─────────────────────────────────────────────────────────────

def annotation_path(name, filename):
    return annotations_dir(name) / f'{safe_filename(filename)}.json'


def read_annotation(name, filename):
    data = read_json(annotation_path(name, filename))
    if not isinstance(data, dict):
        return {'filename': filename, 'regions': [], 'annotated': False}
    data.setdefault('filename', filename)
    # Elements are filtered, not just the root: a hand-edited or imported file
    # with a string inside "regions" used to raise AttributeError from every
    # code path that walked it, permanently breaking the project's gallery,
    # stats and training with an opaque 500.
    regions = data.get('regions')
    data['regions'] = [r for r in regions if isinstance(r, dict)] if isinstance(regions, list) else []
    data.setdefault('annotated', bool(data['regions']))
    return data


def write_annotation(name, filename, data):
    """
    Persist one annotation file.

    Deliberately lets AtomicWriteError propagate: annotations are the user's
    work, and reporting a save that did not happen is worse than an error.
    """
    try:
        write_json(annotation_path(name, filename), data)
    except AtomicWriteError as exc:
        raise ProjectError(
            f'Could not save annotations for {filename}: {exc}', status=503
        ) from exc


def _ensure_dimensions(name, filename, ann):
    """
    Make sure the annotation record carries the image's pixel dimensions.

    Training needs them to normalise boxes. They used to be missing entirely,
    which silently produced unusable labels, so they are cached here on first
    access and reused from then on.
    """
    if ann.get('width') and ann.get('height'):
        return ann
    size = image_size(images_dir(name) / filename)
    if not size:
        return ann
    ann['width'], ann['height'] = size
    write_annotation(name, filename, ann)
    return ann


def save_annotations(name, filename, regions):
    _require(name)
    filename = safe_filename(filename)
    img_path = images_dir(name) / filename
    if not img_path.exists():
        raise ProjectError('Image not found', status=404)

    size = image_size(img_path)
    img_w, img_h = size if size else (None, None)

    clean = []
    for region in regions or []:
        try:
            tag = str(region.get('tag', '')).strip()
            x = float(region.get('x') or 0)
            y = float(region.get('y') or 0)
            w = float(region.get('width') or 0)
            h = float(region.get('height') or 0)
        except (TypeError, ValueError):
            continue
        if not tag:
            continue
        if any(math.isnan(v) or math.isinf(v) for v in (x, y, w, h)):
            continue
        # Clamp to the image so exporters never emit an out-of-bounds box.
        if img_w and img_h:
            x2, y2 = min(x + w, img_w), min(y + h, img_h)
            x, y = max(0.0, x), max(0.0, y)
            w, h = x2 - x, y2 - y
        if w < 1 or h < 1:
            continue
        clean.append({'tag': tag, 'x': x, 'y': y, 'width': w, 'height': h})

    existing = read_annotation(name, filename)
    data = {
        'filename': filename,
        'regions': clean,
        'annotated': len(clean) > 0,
        'width': img_w,
        'height': img_h,
        'updated_at': datetime.now().isoformat(),
    }
    # Everything about the image that saving boxes has no business changing.
    #
    # This is a whitelist rather than a merge because the fields above are
    # authoritative -- regions, annotated and the dimensions are exactly what
    # this call is for -- while the rest describe where the image came from and
    # must survive. Drawing a box used to erase the batch number an upload had
    # written, so the next upload saw no batches at all and started again at 1.
    for key in ('augmented', 'augmentation',   # a copy is not re-augmented
                'batch', 'imported_at',        # which upload it arrived in
                'original_name',               # what the file was called
                'auto_labelled', 'auto_label'):  # drawn by a model, for review
        if key in existing:
            data[key] = existing[key]

    write_annotation(name, filename, data)
    _update_index_entry(name, filename, data)
    refresh_stats(name)
    return {'saved_count': len(clean)}


# ── images ──────────────────────────────────────────────────────────────────

def iter_image_files(name):
    directory = images_dir(name)
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_images(name):
    """
    Every image with its annotation state and pixel dimensions.

    Served from the index, so this is one file read rather than one per image.
    Dimensions are included because the dataset builder needs them; leaving
    them out was what made YOLO label conversion produce unusable labels.
    """
    _require(name)

    # With the folder gone there is nothing to show. The cached index is left
    # on disk untouched so nothing is lost when the files come back, but
    # serving it here would fill the gallery with tiles that every one of
    # which fails to load. The project page explains the situation instead.
    if not images_dir(name).is_dir():
        return []

    index = load_index(name)
    entries = index.get('entries', {})
    result = []
    for filename in sorted(entries):
        entry = entries[filename]
        result.append({
            'filename': filename,
            'annotated': bool(entry.get('annotated')),
            'regions_count': entry.get('regions_count', 0),
            'tags': sorted(entry.get('tag_counts', {})),
            'boxes': entry.get('boxes', []),
            'width': entry.get('width'),
            'height': entry.get('height'),
            'augmented': bool(entry.get('augmented')),
            # Which upload this arrived in, so the gallery can separate them
            # and auto-labelling can be pointed at one. None for images from
            # before uploads were numbered, which reads as the original set.
            'batch': entry.get('batch'),
            'imported_at': entry.get('imported_at'),
            'size_kb': entry.get('size_kb'),
        })
    return result


def get_image_data(name, filename):
    """Image bytes as a data URL plus its annotations, for the annotator."""
    import base64

    import cv2

    _require(name)
    filename = safe_filename(filename)
    img_path = images_dir(name) / filename
    if not img_path.exists():
        raise ProjectError('Image not found', status=404)

    img = imread(img_path)
    if img is None:
        raise ProjectError('Image file is corrupt or unreadable', status=422)

    ok, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ProjectError('Failed to encode image', status=500)

    ann = _ensure_dimensions(name, filename, read_annotation(name, filename))
    return {
        'filename': filename,
        'image': 'data:image/jpeg;base64,' + base64.b64encode(buffer.tobytes()).decode('ascii'),
        'width': int(img.shape[1]),
        'height': int(img.shape[0]),
        'annotations': ann,
    }


def next_batch_number(name):
    """
    The number the next upload will be recorded under.

    Once a project is being extended rather than built -- a second site, a new
    shift, a run of parts that the model got wrong -- "which images are the new
    ones" stops being obvious from the file list. Each upload is stamped with a
    number so the gallery can separate them and auto-labelling can be pointed
    at the newest set rather than at everything unlabelled.
    """
    highest = 0
    for entry in load_index(name).get('entries', {}).values():
        try:
            highest = max(highest, int(entry.get('batch') or 0))
        except (TypeError, ValueError):
            continue
    return highest + 1


def import_images(name, file_objects):
    """Store uploaded files under images/ with collision-free names."""
    _require(name)
    target_dir = images_dir(name)
    target_dir.mkdir(parents=True, exist_ok=True)

    batch = next_batch_number(name)
    imported_at = datetime.now().isoformat()

    imported, rejected = [], []
    for index, file in enumerate(file_objects):
        original = getattr(file, 'filename', '') or ''
        ext = Path(original).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            rejected.append({'filename': original, 'reason': 'unsupported_extension'})
            continue

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        new_name = f'{stamp}_{index:04d}{ext}'
        try:
            file.save(str(target_dir / new_name))
        except Exception as exc:  # noqa: BLE001 - report, keep importing the rest
            rejected.append({'filename': original, 'reason': str(exc)})
            continue

        size = image_size(target_dir / new_name)
        if size is None:
            (target_dir / new_name).unlink(missing_ok=True)
            rejected.append({'filename': original, 'reason': 'not_a_valid_image'})
            continue

        write_annotation(name, new_name, {
            'filename': new_name,
            'regions': [],
            'annotated': False,
            'width': size[0],
            'height': size[1],
            'original_name': original,
            'batch': batch,
            'imported_at': imported_at,
        })
        imported.append({'filename': new_name, 'original_name': original,
                         'batch': batch})

    # One rebuild after a bulk import beats patching the index per file.
    rebuild_index(name)
    refresh_stats(name)
    return {'imported': imported, 'rejected': rejected,
            'imported_count': len(imported), 'batch': batch,
            'imported_at': imported_at}


def delete_image(name, filename):
    _require(name)
    filename = safe_filename(filename)
    image_file = images_dir(name) / filename
    _remove_with_retry(lambda: image_file.unlink(missing_ok=True), filename)
    annotation_path(name, filename).unlink(missing_ok=True)
    _remove_index_entry(name, filename)
    refresh_stats(name)
    return {'message': f'{filename} deleted'}


def delete_images(name, filenames=None, only_generated=False):
    """
    Delete several images in one pass.

    One at a time each delete rebuilt the counters and rewrote the index, which
    on a large project makes clearing a few hundred generated copies slower
    than generating them was. Here the removals happen first and the
    bookkeeping once at the end.

    `only_generated` deletes every filtered copy in the project, which is the
    common case: a filter preset turned out to suit the data badly and the
    hundred images it wrote are all in the way.
    """
    _require(name)

    if only_generated:
        targets = [entry['filename'] for entry in list_images(name)
                   if entry.get('augmented')]
    else:
        targets = []
        for filename in filenames or []:
            # One bad name should not abort the batch; the caller is told what
            # was refused rather than losing the whole request.
            try:
                targets.append(safe_filename(filename))
            except ProjectError:
                continue

    folder = images_dir(name)
    removed, failed = [], []
    for filename in targets:
        try:
            _remove_with_retry(lambda f=filename: (folder / f).unlink(missing_ok=True),
                               filename)
            annotation_path(name, filename).unlink(missing_ok=True)
            removed.append(filename)
        except ProjectError as exc:
            failed.append({'filename': filename, 'reason': exc.message})
        except OSError as exc:
            failed.append({'filename': filename, 'reason': str(exc)})

    if removed:
        # Cheaper than removing each entry in turn once the list is long.
        rebuild_index(name)
    summary = refresh_stats(name)

    message = f'Deleted {len(removed)} image(s)'
    if failed:
        message += f'; {len(failed)} could not be removed'
    return {
        'message': message,
        'deleted_count': len(removed),
        'deleted': removed[:200],
        'failed': failed[:50],
        'total_images': summary.get('total_images'),
        'annotated_images': summary.get('annotated_images'),
    }


# ── statistics ──────────────────────────────────────────────────────────────

def refresh_stats(name):
    """
    Recompute the cached counters and persist them.

    Derived from the gallery index rather than by re-reading every annotation
    file, which is what let an annotator's save take half a second on a
    2000-image project.

    If the images directory is missing altogether the counters are left
    untouched. A missing directory means the data is not reachable — an
    unmounted drive, a mistyped PROJECTS_ROOT, a partial checkout — and is not
    the same thing as a project whose images were deleted. Recomputing then
    would silently overwrite a real dataset's statistics with zeros.
    """
    _require(name)

    if not images_dir(name).is_dir():
        return _read_meta(name)

    entries = load_index(name).get('entries', {})

    tags = {}
    total_boxes = 0
    annotated_images = 0

    for entry in entries.values():
        if entry.get('annotated'):
            annotated_images += 1
        total_boxes += entry.get('regions_count', 0)
        for tag, count in (entry.get('tag_counts') or {}).items():
            stats = tags.setdefault(tag, {'boxes': 0, 'images': 0})
            stats['boxes'] += count
            stats['images'] += 1

    meta = _read_meta(name)
    meta['total_images'] = len(entries)
    meta['annotated_images'] = annotated_images
    meta['total_annotations'] = total_boxes
    # Sorted so the class index assigned during training is stable between runs.
    meta['tags'] = {tag: tags[tag] for tag in sorted(tags)}
    return _write_meta(name, meta)


def rescan(name, clear_if_missing=False):
    """
    Re-read the project from disk and rebuild its caches.

    Two situations need this. Files that were restored after the app started
    should appear without a restart. And a project whose files are genuinely
    gone is otherwise stuck reporting counts from a cache that can never be
    recomputed, because refresh_stats deliberately refuses to zero a project
    it cannot see — that guard protects an unmounted drive, but it also means
    the only way out is to ask explicitly.
    """
    _require(name)
    present = images_dir(name).is_dir()

    if present:
        rebuild_index(name)
        meta = refresh_stats(name)
        return {
            'images_available': True,
            'total_images': meta['total_images'],
            'annotated_images': meta['annotated_images'],
            'total_annotations': meta['total_annotations'],
            'message': f'Found {meta["total_images"]} images.',
        }

    if not clear_if_missing:
        return {
            'images_available': False,
            'total_images': _read_meta(name).get('total_images', 0),
            'message': (
                f'Still nothing readable at {images_dir(name)}. The counts have '
                'been left alone in case the files are only temporarily '
                'unavailable.'
            ),
        }

    # Explicitly asked for: the files are not coming back, so the counters are
    # reset to match what is actually there.
    meta = _read_meta(name)
    meta.update({'total_images': 0, 'annotated_images': 0,
                 'total_annotations': 0, 'tags': {}})
    _write_meta(name, meta)
    index_path(name).unlink(missing_ok=True)
    return {
        'images_available': False,
        'total_images': 0,
        'message': 'Counts cleared. Import images to start this project again.',
    }


def class_names(name):
    """Class list in the exact order used to assign training class indices."""
    return sorted(_read_meta(name).get('tags', {}).keys())


def dataset_summary(name):
    """Dataset statistics plus a 0-100 readiness score and concrete advice."""
    meta = refresh_stats(name)

    total_images = meta['total_images']
    annotated_images = meta['annotated_images']
    total_boxes = meta['total_annotations']
    tags = meta['tags']

    score = 0.0
    warnings = []

    # Coverage — how much of the dataset is labelled (30%).
    if total_images:
        coverage = annotated_images / total_images
        score += min(coverage, 1.0) * 100 * 0.30
        if coverage < 0.5:
            warnings.append(
                f'Only {annotated_images}/{total_images} images are annotated. '
                'Aim for at least 50%.'
            )
    else:
        warnings.append('No images in this project yet.')

    # Per-class support — the weakest class caps model quality (40%).
    MIN_PER_CLASS = 30
    if tags:
        counts = {tag: stats['images'] for tag, stats in tags.items()}
        per_class = list(counts.values())
        weakest = min(per_class)
        score += min(weakest / MIN_PER_CLASS, 1.0) * 100 * 0.40

        # Named, and sorted by how short they are. "The smallest class has 2
        # images" leaves someone with nine classes no idea which to go and
        # photograph; the point of the warning is to say what to do next.
        short = sorted((c for c in counts.items() if c[1] < MIN_PER_CLASS),
                       key=lambda item: item[1])
        if short:
            listed = ', '.join(f'{tag} ({count})' for tag, count in short[:6])
            more = f' and {len(short) - 6} more' if len(short) > 6 else ''
            warnings.append(
                f'{len(short)} class(es) below {MIN_PER_CLASS} images: '
                f'{listed}{more}.'
            )
        if len(tags) > 1 and max(per_class) > weakest * 10:
            biggest = max(counts.items(), key=lambda item: item[1])
            smallest = min(counts.items(), key=lambda item: item[1])
            warnings.append(
                f'Classes are heavily imbalanced: "{biggest[0]}" has '
                f'{biggest[1]} images against {smallest[1]} for '
                f'"{smallest[0]}". A model trained on this will favour the '
                'common ones.'
            )
    else:
        warnings.append('No classes defined — draw and tag some boxes first.')

    # Box density — very sparse labelling usually means missed objects (30%).
    if annotated_images:
        density = total_boxes / annotated_images
        score += min(density / 3.0, 1.0) * 100 * 0.30
        if density < 0.5:
            warnings.append(
                'Very low annotation density. Check that every visible object '
                'is labelled.'
            )

    # A validation split needs at least a couple of images to be meaningful.
    if 0 < annotated_images < 10:
        warnings.append(
            f'Only {annotated_images} annotated images. Training will run but '
            'validation metrics will not be meaningful below ~10 images.'
        )

    score = round(score, 1)
    recommendations = []
    if score < 40:
        recommendations.append('Not ready to train. Add more annotated images.')
    elif score < 70:
        recommendations.append('Training will run, but accuracy will be limited.')
    else:
        recommendations.append('Dataset looks ready for training.')
    if tags:
        counts = {tag: stats['images'] for tag, stats in tags.items()}
        short = sorted((c for c in counts.items() if c[1] < MIN_PER_CLASS),
                       key=lambda item: item[1])
        if short:
            worst = short[0]
            recommendations.append(
                f'Photograph more examples of "{worst[0]}" first: it has '
                f'{worst[1]} and wants about {MIN_PER_CLASS - worst[1]} more.'
            )

        # A small dataset and a large model is the classic way to get a
        # confident model that has memorised the training set. Say so before
        # the run rather than after it.
        if annotated_images < 60:
            recommendations.append(
                f'With {annotated_images} annotated images, use a small model '
                '(yolo11n or yolo11s). A large one will memorise this set '
                'rather than learn from it.'
            )
        if 0 < annotated_images < 200:
            recommendations.append(
                'Few images benefit from more passes: 150-300 epochs is a '
                'better starting point than 100.'
            )
    if annotated_images < 100:
        recommendations.append('For production quality aim for 100+ images per class.')

    return {
        'total_images': total_images,
        'annotated_images': annotated_images,
        'total_boxes': total_boxes,
        'images_available': images_dir(name).is_dir(),
        'images_path': str(images_dir(name)),
        'tags': tags,
        'classes': sorted(tags),
        'num_classes': len(tags),
        'readiness_score': score,
        'warnings': warnings,
        'recommendations': recommendations,
    }
