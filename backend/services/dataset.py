"""
Dataset preparation, export and import.

build_yolo_dataset() converts the project's JSON annotations into the
images/ + labels/ + data.yaml layout ultralytics expects. This is the step that
decides whether training produces a usable model, so it validates its own
output instead of writing whatever it can and letting the trainer fail later.
"""

import json
import os
import random
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

from config import IMAGE_EXTENSIONS
from services import projects
from services.imaging import image_size
from services.projects import ProjectError

VAL_FRACTION = 0.2
MIN_VAL_IMAGES = 1

# Import limits. A zip's declared sizes cannot be trusted, so the uncompressed
# total is capped as members are written: a few-megabyte archive can otherwise
# expand until the disk is full.
MAX_IMPORT_BYTES = 8 * 1024 * 1024 * 1024   # 8 GB total
MAX_MEMBER_BYTES = 256 * 1024 * 1024        # 256 MB per image


def _link_or_copy(src: Path, dst: Path):
    """
    Hard-link the image into the dataset folder, falling back to a copy.

    A project can hold thousands of images and the dataset is rebuilt on every
    training run; hard links make that near-instant and use no extra disk.
    """
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(src, dst)


def _group_key(entry, annotation):
    """
    Images derived from the same original belong to the same split.

    An augmented copy is a near-duplicate of its source. If the two land on
    opposite sides of the train/val split, validation metrics measure
    memorisation rather than generalisation.
    """
    augmentation = annotation.get('augmentation') or {}
    return augmentation.get('source_image') or entry['filename']


def _resolve_size(name, entry, annotation):
    width = entry.get('width') or annotation.get('width')
    height = entry.get('height') or annotation.get('height')
    if width and height:
        return int(width), int(height)
    size = image_size(projects.images_dir(name) / entry['filename'])
    return size if size else None


def build_yolo_dataset(name, seed=42):
    """
    Write <project>/training/dataset/ and return a report about it.

    Raises ProjectError when the result would not be trainable, so the caller
    can surface a real reason instead of starting a run that dies minutes later.
    """
    projects.get_project(name)
    # Training is the one place a stale index would be costly, so the index is
    # rebuilt from the annotation files before the split is computed.
    projects.rebuild_index(name)
    classes = projects.class_names(name)
    if not classes and projects.images_dir(name).is_dir():
        raise ProjectError('No classes defined. Tag some boxes before training.')
    class_to_id = {cls: index for index, cls in enumerate(classes)}

    all_entries = projects.list_images(name)
    entries = [e for e in all_entries if e['annotated']]
    if not entries:
        # Distinguish the three ways this happens; "no annotated images" is
        # actively misleading when the image files are not on disk at all.
        if not projects.images_dir(name).is_dir():
            raise ProjectError(
                f'The images folder for "{name}" does not exist '
                f'({projects.images_dir(name)}). If the dataset lives '
                'elsewhere, point PROJECTS_ROOT at it.'
            )
        if not all_entries:
            raise ProjectError(
                f'Project "{name}" has no images. Import some before training.'
            )
        raise ProjectError(
            f'None of the {len(all_entries)} images in "{name}" are annotated yet.'
        )

    annotations = {e['filename']: projects.read_annotation(name, e['filename'])
                   for e in entries}

    # ── split by source group, keeping augmented copies out of validation ──
    groups = {}
    for entry in entries:
        groups.setdefault(_group_key(entry, annotations[entry['filename']]), []).append(entry)

    group_keys = sorted(groups)
    random.Random(seed).shuffle(group_keys)

    val_target = max(MIN_VAL_IMAGES, int(round(len(group_keys) * VAL_FRACTION)))
    val_target = min(val_target, max(0, len(group_keys) - 1))

    val_keys = set(group_keys[:val_target])
    train_keys = set(group_keys[val_target:])

    split_entries = {'train': [], 'val': []}
    held_back = []
    for key in group_keys:
        for entry in groups[key]:
            if key in train_keys:
                split_entries['train'].append(entry)
            elif entry['augmented']:
                # A generated copy of a validation image belongs in neither
                # split. Training on it leaks the validation image into
                # training, and validating on it scores the model against
                # synthetic near-duplicates, which inflates the number.
                #
                # This used to happen silently: someone who generated thirty
                # copies and got a dataset six images smaller than the folder
                # had nothing to tell them where they went.
                held_back.append(entry['filename'])
                continue
            else:
                split_entries['val'].append(entry)

    # Degenerate datasets (one group, or a group that held only augmented
    # copies) still need both splits populated for training to run at all.
    if not split_entries['train']:
        split_entries['train'] = [e for key in group_keys for e in groups[key]]
    if not split_entries['val']:
        originals = [e for e in split_entries['train'] if not e['augmented']]
        pool = originals or split_entries['train']
        split_entries['val'] = pool[:max(1, len(pool) // 5)]

    # ── write the dataset ─────────────────────────────────────────────────
    dataset_dir = projects.training_dir(name) / 'dataset'
    # A stale dataset from a previous run would leave labels for images and
    # classes that no longer exist, so it is rebuilt from scratch every time.
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ('train', 'val'):
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    images_dir = projects.images_dir(name)
    counts = {'train': 0, 'val': 0}
    boxes = {'train': 0, 'val': 0}
    per_class = {cls: 0 for cls in classes}
    skipped = []

    for split, split_list in split_entries.items():
        for entry in split_list:
            filename = entry['filename']
            source = images_dir / filename
            if not source.exists():
                skipped.append({'filename': filename, 'reason': 'missing_file'})
                continue

            size = _resolve_size(name, entry, annotations[filename])
            if not size:
                skipped.append({'filename': filename, 'reason': 'unknown_dimensions'})
                continue
            img_w, img_h = size
            if img_w <= 0 or img_h <= 0:
                skipped.append({'filename': filename, 'reason': 'invalid_dimensions'})
                continue

            lines = []
            for region in annotations[filename].get('regions', []):
                tag = str(region.get('tag') or '').strip()
                if tag not in class_to_id:
                    continue
                try:
                    x = float(region['x'])
                    y = float(region['y'])
                    w = float(region['width'])
                    h = float(region['height'])
                except (KeyError, TypeError, ValueError):
                    continue

                # Clamp to the image, then convert to normalised centre form.
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(img_w), x + w), min(float(img_h), y + h)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue

                cx = ((x1 + x2) / 2.0) / img_w
                cy = ((y1 + y2) / 2.0) / img_h
                nw = (x2 - x1) / img_w
                nh = (y2 - y1) / img_h
                if not (0 < nw <= 1 and 0 < nh <= 1):
                    continue
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)

                lines.append(f'{class_to_id[tag]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')
                per_class[tag] += 1

            if not lines:
                skipped.append({'filename': filename, 'reason': 'no_valid_boxes'})
                continue

            _link_or_copy(source, dataset_dir / 'images' / split / filename)
            label_file = dataset_dir / 'labels' / split / f'{Path(filename).stem}.txt'
            label_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')

            counts[split] += 1
            boxes[split] += len(lines)

    if counts['train'] == 0:
        raise ProjectError(
            'No usable training labels were produced. Check that annotation '
            'boxes lie inside their images and that images are readable.'
        )
    if counts['val'] == 0:
        raise ProjectError('No usable validation labels were produced.')

    data_yaml = dataset_dir / 'data.yaml'
    with open(data_yaml, 'w', encoding='utf-8') as f:
        yaml.safe_dump(
            {
                'path': str(dataset_dir.resolve()),
                'train': 'images/train',
                'val': 'images/val',
                'nc': len(classes),
                # Explicit index -> name mapping; a bare list is order-sensitive
                # and easy to get wrong when classes are added later.
                'names': {index: cls for index, cls in enumerate(classes)},
            },
            f, allow_unicode=True, sort_keys=False,
        )

    empty_classes = [cls for cls, count in per_class.items() if count == 0]
    return {
        'dataset_path': str(dataset_dir.resolve()),
        'data_yaml': str(data_yaml.resolve()),
        'classes': classes,
        'train_images': counts['train'],
        'val_images': counts['val'],
        'train_boxes': boxes['train'],
        'val_boxes': boxes['val'],
        'boxes_per_class': per_class,
        'empty_classes': empty_classes,
        'skipped': skipped[:100],
        'skipped_count': len(skipped),
        'held_back_count': len(held_back),
        'held_back': held_back[:50],
        'held_back_reason': (
            f'{len(held_back)} generated copies were left out because their '
            'source image is in the validation set: training on them would '
            'leak it, and validating on them would flatter the score.'
        ) if held_back else '',
    }


# ── zip export / import ─────────────────────────────────────────────────────

def export_dataset(name):
    """Bundle images and annotations into a single zip under exports/."""
    projects.get_project(name)
    export_dir = projects.exports_dir(name)
    export_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive = export_dir / f'{name}_dataset_{stamp}.zip'
    summary = projects.refresh_stats(name)

    manifest = {
        'project_name': name,
        'export_date': datetime.now().isoformat(),
        'total_images': summary['total_images'],
        'total_annotations': summary['total_annotations'],
        'tags': summary['tags'],
        'annotations': [],
    }

    image_files = projects.iter_image_files(name)
    for image in image_files:
        manifest['annotations'].append(projects.read_annotation(name, image.name))

    # Streamed straight into the zip — a project can be several GB and the old
    # code copied the whole image tree to a temp folder first.
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr('dataset.json', json.dumps(manifest, indent=2, ensure_ascii=False))
        for image in image_files:
            zf.write(image, f'images/{image.name}')

    return {'export_file': str(archive), 'export_name': archive.name,
            'image_count': len(image_files)}


def import_dataset(name, zip_path):
    """Add the images and annotations from an exported zip to this project."""
    projects.get_project(name)
    images_dir = projects.images_dir(name)
    images_dir.mkdir(parents=True, exist_ok=True)

    imported_images = 0
    imported_boxes = 0
    total_written = 0
    skipped = []

    with zipfile.ZipFile(zip_path) as zf:
        try:
            manifest = json.loads(zf.read('dataset.json').decode('utf-8'))
        except KeyError:
            raise ProjectError('Not a valid dataset export: dataset.json is missing')
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProjectError('Not a valid dataset export: dataset.json is unreadable')

        names_in_zip = set(zf.namelist())
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for index, annotation in enumerate(manifest.get('annotations', [])):
            original = annotation.get('filename')
            if not original:
                continue
            member = f'images/{Path(original).name}'
            if member not in names_in_zip:
                continue

            ext = Path(original).suffix.lower()
            # Checked before anything is written. Without this, a member with
            # no extension (or .php) landed in images/, was counted as
            # imported, and was then permanently invisible because the gallery
            # only lists IMAGE_EXTENSIONS.
            if ext not in IMAGE_EXTENSIONS:
                skipped.append({'filename': original, 'reason': 'unsupported_extension'})
                continue

            new_name = f'{stamp}_{index:05d}{ext}'
            target = images_dir / new_name
            # Read the member by name rather than extracting the archive, so a
            # crafted zip cannot write outside the project directory. Copied in
            # chunks with a running total so a zip bomb is stopped mid-write
            # rather than after it has filled the disk.
            written = 0
            aborted = False
            with zf.open(member) as src, open(target, 'wb') as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    total_written += len(chunk)
                    if written > MAX_MEMBER_BYTES or total_written > MAX_IMPORT_BYTES:
                        aborted = True
                        break
                    dst.write(chunk)

            if aborted:
                target.unlink(missing_ok=True)
                skipped.append({'filename': original, 'reason': 'too_large'})
                if total_written > MAX_IMPORT_BYTES:
                    break
                continue

            size = image_size(target)
            if size is None:
                target.unlink(missing_ok=True)
                skipped.append({'filename': original, 'reason': 'not_a_valid_image'})
                continue

            # Sanitised the same way a save from the annotator would be; an
            # imported file is no more trustworthy than a hand-edited one.
            regions = [r for r in (annotation.get('regions') or [])
                       if isinstance(r, dict)]
            projects.write_annotation(name, new_name, {
                'filename': new_name,
                'regions': regions,
                'annotated': bool(annotation.get('annotated') and regions),
                'width': size[0],
                'height': size[1],
                'augmented': bool(annotation.get('augmented')),
                'augmentation': annotation.get('augmentation'),
                'imported_from': original,
            })
            imported_images += 1
            imported_boxes += len(regions)

    projects.rebuild_index(name)
    projects.refresh_stats(name)
    message = f'Imported {imported_images} images with {imported_boxes} annotations'
    if skipped:
        message += f'; skipped {len(skipped)}'
    return {
        'message': message,
        'imported_images': imported_images,
        'imported_annotations': imported_boxes,
        'skipped': skipped[:50],
        'skipped_count': len(skipped),
    }
