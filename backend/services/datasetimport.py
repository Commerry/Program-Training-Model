"""
Bring in an annotated dataset that was labelled somewhere else.

Six thousand pictures already labelled in Custom Vision are worth more than
any model exported from it: a model is a frozen answer, while the pictures and
their boxes can be trained again, augmented, corrected and extended. Retyping
them is not an option and neither is uploading them one at a time through a
browser, so this reads the export folder where it sits.

Three layouts are recognised, because the tools people arrive from write
different ones:

  yolo   images/ beside labels/, one .txt per image holding
         `class cx cy w h` normalised, and a label.txt naming the classes in
         index order. This is what Custom Vision writes.
  coco   one .json with images, annotations and categories, boxes as
         [x, y, width, height] in pixels.
  voc    one .xml per image, boxes as xmin/ymin/xmax/ymax in pixels.

The class names come from the export and are used in the order the export
gives them. That order is not alphabetical and must not be sorted: a YOLO file
says `8`, and only label.txt knows that 8 is Good. Sorting the names would
relabel every box in the set with a different word while looking entirely
correct.

Reading runs in a background thread with a progress file, the same as training
and auto-labelling: six thousand images is minutes of work, and a request that
runs for minutes is a request that times out.
"""

import json
import shutil
import threading
import time
import xml.etree.ElementTree as ElementTree
from datetime import datetime
from pathlib import Path

from services import projects
from services.atomicio import read_json, write_json_best_effort
from services.projects import ProjectError

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Files an exporter uses to name its classes, in the order worth trying.
CLASS_FILES = ('label.txt', 'labels.txt', 'classes.txt', 'obj.names',
               'predefined_classes.txt')

_locks = {}
_locks_guard = threading.Lock()


def _lock(name):
    with _locks_guard:
        return _locks.setdefault(name, threading.Lock())


def status_path(name):
    return projects.training_dir(name) / 'dataset_import_status.json'


def get_status(name):
    projects.get_project(name)
    return read_json(status_path(name))


def _write_status(name, **fields):
    status = read_json(status_path(name)) or {}
    status.update(fields)
    write_json_best_effort(status_path(name), status)
    return status


def _folder(raw_path):
    if not raw_path or not str(raw_path).strip():
        raise ProjectError('Give the folder the dataset was exported to')
    folder = Path(str(raw_path).strip().strip('"')).expanduser()
    if not folder.is_dir():
        raise ProjectError(f'No such folder: {folder}')
    return folder


def _image_files(folder):
    return sorted(p for p in folder.rglob('*')
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _class_names(folder):
    """
    The class names in the order the export gives them, never sorted.

    A YOLO annotation says `8`. Only this file knows that 8 is Good, and it is
    not in alphabetical order -- sorting it would put a different word on every
    box in the set while looking entirely correct.
    """
    for candidate in CLASS_FILES:
        for path in (folder / candidate, folder.parent / candidate):
            if path.is_file():
                names = [line.strip() for line in
                         path.read_text(encoding='utf-8-sig').splitlines()]
                names = [n for n in names if n]
                if names:
                    return names, path.name

    # data.yaml, as ultralytics writes it. Read without pulling in a YAML
    # parser: the names are either a list or an index map, and both are
    # recognisable line by line.
    for path in (folder / 'data.yaml', folder.parent / 'data.yaml'):
        if not path.is_file():
            continue
        names, collecting = [], False
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            stripped = line.strip()
            if stripped.startswith('names:'):
                collecting = True
                inline = stripped.split(':', 1)[1].strip()
                if inline.startswith('[') and inline.endswith(']'):
                    return ([n.strip().strip('\'"') for n in
                             inline[1:-1].split(',') if n.strip()], path.name)
                continue
            if collecting:
                if stripped.startswith('- '):
                    names.append(stripped[2:].strip().strip('\'"'))
                elif ':' in stripped and stripped.split(':')[0].strip().isdigit():
                    names.append(stripped.split(':', 1)[1].strip().strip('\'"'))
                elif stripped and not stripped.startswith('#'):
                    break
        if names:
            return names, path.name

    return [], None


def detect(raw_path):
    """
    What is in this folder, without changing anything.

    Answering "is this the right folder and will it come in whole" before
    copying six thousand files is worth the pass it costs.
    """
    folder = _folder(raw_path)

    images = _image_files(folder)
    if not images:
        raise ProjectError(f'No images anywhere under {folder}')

    names, names_from = _class_names(folder)

    coco = [p for p in folder.glob('*.json')] + [p for p in folder.glob('*/*.json')]
    for candidate in coco:
        try:
            data = json.loads(candidate.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and 'annotations' in data and 'images' in data:
            categories = data.get('categories') or []
            return {
                'format': 'coco',
                'folder': str(folder),
                'annotation_file': candidate.name,
                'images': len(data.get('images') or []),
                'boxes': len(data.get('annotations') or []),
                'classes': [str(c.get('name')) for c in categories],
                'classes_from': candidate.name,
                'image_files': len(images),
            }

    label_dirs = [d for d in (folder / 'labels', folder / 'labels_train',
                              folder) if d.is_dir()]
    text_files = []
    for directory in label_dirs:
        text_files = [p for p in directory.glob('*.txt')
                      if p.name not in CLASS_FILES]
        if text_files:
            break

    if text_files:
        boxes, malformed = 0, 0
        used = set()
        for path in text_files[:400]:
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    malformed += 1
                    continue
                boxes += 1
                try:
                    used.add(int(parts[0]))
                except ValueError:
                    malformed += 1
        # Sampled, so scale the count rather than claiming a total nobody read.
        estimated = round(boxes * len(text_files) / max(len(text_files[:400]), 1))
        return {
            'format': 'yolo',
            'folder': str(folder),
            'annotation_file': str(text_files[0].parent.name) + '/',
            'images': len(text_files),
            'boxes': estimated,
            'classes': names,
            'classes_from': names_from,
            'image_files': len(images),
            'highest_class_seen': max(used) if used else None,
            'malformed': malformed,
        }

    xml_files = [p for p in folder.rglob('*.xml')]
    if xml_files:
        return {
            'format': 'voc',
            'folder': str(folder),
            'annotation_file': xml_files[0].parent.name + '/',
            'images': len(xml_files),
            'boxes': None,
            'classes': names,
            'classes_from': names_from,
            'image_files': len(images),
        }

    raise ProjectError(
        f'{len(images)} image(s) in {folder}, but no annotations beside them. '
        'Expected a labels folder of .txt files (YOLO), a .json (COCO), or '
        '.xml files (Pascal VOC).')


def preview(raw_path):
    """What detect found, plus what would go wrong if it were imported."""
    found = detect(raw_path)
    warnings = []

    if found['format'] == 'yolo':
        if not found['classes']:
            warnings.append(
                'No label.txt naming the classes, so the boxes would come in '
                'as numbers. The names cannot be recovered later from the '
                'files alone.')
        elif found.get('highest_class_seen') is not None and \
                found['highest_class_seen'] >= len(found['classes']):
            warnings.append(
                f'A box refers to class {found["highest_class_seen"]} but only '
                f'{len(found["classes"])} names were found. The wrong file is '
                'being read, or it is incomplete.')
        if found.get('malformed'):
            warnings.append(f'{found["malformed"]} line(s) are not '
                            '"class cx cy w h" and would be skipped.')

    if found['image_files'] != found['images']:
        warnings.append(
            f'{found["image_files"]} image file(s) but {found["images"]} '
            'annotation(s). Images with no annotation come in unlabelled.')

    found['warnings'] = warnings
    return found


# ── reading each layout ─────────────────────────────────────────────────────
# Each returns {image stem: [{'tag', 'x', 'y', 'width', 'height'}]} in pixels.


def _read_yolo(folder, names):
    directory = next((d for d in (folder / 'labels', folder / 'labels_train',
                                  folder) if d.is_dir()
                      and any(p.name not in CLASS_FILES
                              for p in d.glob('*.txt'))), folder)
    per_image = {}
    for path in directory.glob('*.txt'):
        if path.name in CLASS_FILES:
            continue
        per_image[path.stem] = path
    return per_image, 'yolo', names


def _yolo_regions(path, names, width, height):
    regions = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            index = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            continue
        tag = names[index] if 0 <= index < len(names) else f'class_{index}'
        # Normalised centre and size, in a frame this file never states: the
        # image is the only place the pixel dimensions come from.
        x = (cx - w / 2) * width
        y = (cy - h / 2) * height
        box_w, box_h = w * width, h * height
        if box_w < 1 or box_h < 1:
            continue
        regions.append({'tag': str(tag), 'x': max(0.0, x), 'y': max(0.0, y),
                        'width': box_w, 'height': box_h})
    return regions


def _read_coco(path):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    categories = {c['id']: str(c.get('name') or c['id'])
                  for c in (data.get('categories') or [])}
    files = {img['id']: img for img in (data.get('images') or [])}

    per_image = {}
    for note in (data.get('annotations') or []):
        image = files.get(note.get('image_id'))
        if not image:
            continue
        box = note.get('bbox') or []
        if len(box) != 4:
            continue
        x, y, w, h = (float(v) for v in box)
        if w < 1 or h < 1:
            continue
        stem = Path(str(image.get('file_name') or '')).stem
        per_image.setdefault(stem, []).append({
            'tag': categories.get(note.get('category_id'),
                                  f'class_{note.get("category_id")}'),
            'x': x, 'y': y, 'width': w, 'height': h,
        })
    return per_image, [categories[k] for k in sorted(categories)]


def _read_voc(path):
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError):
        return []
    regions = []
    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        bnd = obj.find('bndbox')
        if not name or bnd is None:
            continue
        try:
            x1 = float(bnd.findtext('xmin'))
            y1 = float(bnd.findtext('ymin'))
            x2 = float(bnd.findtext('xmax'))
            y2 = float(bnd.findtext('ymax'))
        except (TypeError, ValueError):
            continue
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        regions.append({'tag': name, 'x': x1, 'y': y1,
                        'width': x2 - x1, 'height': y2 - y1})
    return regions


# ── the import itself ───────────────────────────────────────────────────────


def start(name, folder_path, limit=None):
    """Copy the dataset into the project in the background."""
    projects.get_project(name)
    found = detect(folder_path)

    lock = _lock(name)
    if not lock.acquire(blocking=False):
        raise ProjectError('An import is already running for this project')

    _write_status(name, status='running', format=found['format'],
                  folder=found['folder'], total=found['images'], done=0,
                  imported=0, skipped=0, boxes=0, started_at=datetime.now().isoformat(),
                  message=f'Reading {found["images"]} annotation(s)')

    thread = threading.Thread(target=_run, args=(name, found, limit, lock),
                              daemon=True)
    thread.start()
    return get_status(name)


def _run(name, found, limit, lock):
    try:
        _import_all(name, found, limit)
    except Exception as exc:  # noqa: BLE001 - reported, never raised into a thread
        _write_status(name, status='failed',
                      message=f'{type(exc).__name__}: {str(exc)[:300]}',
                      finished_at=datetime.now().isoformat())
    finally:
        try:
            lock.release()
        except RuntimeError:
            pass


def _import_all(name, found, limit):
    folder = Path(found['folder'])
    target = projects.images_dir(name)
    target.mkdir(parents=True, exist_ok=True)

    images = {p.stem: p for p in _image_files(folder)}
    names = found.get('classes') or []

    if found['format'] == 'coco':
        annotations, names = _read_coco(folder / found['annotation_file'])
        lookup = None
    elif found['format'] == 'voc':
        annotations = {p.stem: p for p in folder.rglob('*.xml')}
        lookup = 'voc'
    else:
        annotations, lookup, names = _read_yolo(folder, names)

    stems = sorted(annotations) if annotations else sorted(images)
    if limit:
        stems = stems[:int(limit)]

    batch = projects.next_batch_number(name)
    imported_at = datetime.now().isoformat()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    done = imported = skipped = boxes = 0
    for index, stem in enumerate(stems):
        if (read_json(status_path(name)) or {}).get('cancel_requested'):
            _write_status(name, status='cancelled', done=done,
                          finished_at=datetime.now().isoformat(),
                          message=f'Stopped after {imported} image(s)')
            return

        source = images.get(stem)
        done += 1
        if source is None:
            skipped += 1
            continue

        # PIL reads the size without decoding the pixels, and reads a path
        # with non-ASCII characters in it -- which OpenCV on Windows does not,
        # and this export lives under a Thai folder name.
        try:
            from PIL import Image
            with Image.open(source) as opened:
                width, height = opened.size
        except Exception:  # noqa: BLE001 - a file that will not open is skipped
            skipped += 1
            continue

        new_name = f'{stamp}_{index:06d}{source.suffix.lower()}'
        try:
            # Copied rather than re-encoded: nothing here needs the pixels
            # changed, and a re-encode would lose a little of every image.
            shutil.copy2(source, target / new_name)
        except OSError:
            skipped += 1
            continue

        if found['format'] == 'coco':
            regions = annotations.get(stem) or []
        elif lookup == 'voc':
            regions = _read_voc(annotations[stem])
        else:
            regions = _yolo_regions(annotations[stem], names, width, height)

        # Clamp to the frame, as saving by hand does.
        clean = []
        for region in regions:
            x = max(0.0, min(region['x'], width - 1.0))
            y = max(0.0, min(region['y'], height - 1.0))
            w = min(region['width'], width - x)
            h = min(region['height'], height - y)
            if w >= 1 and h >= 1:
                clean.append({'tag': region['tag'], 'x': x, 'y': y,
                              'width': w, 'height': h})

        projects.write_annotation(name, new_name, {
            'filename': new_name,
            'regions': clean,
            'annotated': bool(clean),
            'width': width,
            'height': height,
            'original_name': source.name,
            'batch': batch,
            'imported_at': imported_at,
            # Drawn by people in the tool this came from. Not a prediction, so
            # not something to put in the review queue: marking it auto would
            # send six thousand already-correct pictures to be checked.
            'imported_from': {'format': found['format'],
                              'folder': found['folder']},
        })
        imported += 1
        boxes += len(clean)

        if index % 50 == 0:
            _write_status(name, done=done, imported=imported, skipped=skipped,
                          boxes=boxes,
                          message=f'{imported} of {len(stems)} image(s)')

    projects.rebuild_index(name)
    projects.refresh_stats(name)
    _write_status(name, status='finished', done=done, imported=imported,
                  skipped=skipped, boxes=boxes, batch=batch,
                  finished_at=datetime.now().isoformat(),
                  message=f'{imported} image(s), {boxes} box(es), batch {batch}')


def cancel(name):
    projects.get_project(name)
    status = read_json(status_path(name)) or {}
    if status.get('status') != 'running':
        return {'cancelled': False, 'reason': 'nothing running'}
    _write_status(name, cancel_requested=True)
    return {'cancelled': True}


def wait_for_idle(name, timeout=600):
    """For tests and scripts: block until the running import settles."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = read_json(status_path(name)) or {}
        if status.get('status') in ('finished', 'failed', 'cancelled'):
            return status
        time.sleep(0.2)
    return read_json(status_path(name)) or {}
