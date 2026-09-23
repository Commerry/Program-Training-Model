"""
Models brought in from somewhere else, kept with the files they need.

Pre-labelling is worth most on a project that has nothing yet, and that is
exactly when this installation has no model to offer. Somebody arriving with a
detector from a previous system -- an Azure Custom Vision export, a YOLO from
another toolchain -- should be able to point the auto-labeller at it and find
out in one pass whether it is worth correcting its boxes or starting by hand.

An ONNX on its own is not enough to do that with. It carries no class names,
and nothing in it records how it wants to be fed: padded or squashed, RGB or
BGR, 0..1 or 0..255. Those live in the folder the export came from. So an
import is a folder, not a file:

    data/imported/<slug>/model.onnx
                        /labels.txt          the class names, in order
                        /conventions.txt     how to feed it, when it is known
                        /import.json         what this is and where it came from

which also means the runner finds its own sidecars exactly as it would if the
model were still sitting in the export folder it shipped in.
"""

import json
import re
import shutil
import time
from pathlib import Path

from config import IMPORTED_MODELS_DIR
from services.projects import ProjectError

# The same set the rest of the application will load. A .blob is compiled for
# a Myriad chip and cannot run here, so it is not worth importing.
ALLOWED_SUFFIXES = {'.pt', '.pth', '.onnx', '.torchscript'}

MAX_LABEL_BYTES = 256 * 1024

# A zip of an export folder is trusted no further than its own member names.
MAX_UNPACKED_BYTES = 4 * 1024 * 1024 * 1024
MAX_MEMBERS = 20_000


def _extract(zip_path, folder):
    """
    Unpack an export folder safely, and say where the model in it ended up.

    An Azure Custom Vision export is a folder, and zipping it is the obvious
    way to carry it to another machine: model.onnx together with labels.txt
    and metadata_properties.json, which are the two things an ONNX does not
    carry and this application reads from beside it. Unpacking the whole
    folder means those are found exactly as they would have been in place.

    Members are rebuilt from their parts rather than extracted wholesale,
    because extractall will happily write ../../anywhere.
    """
    import zipfile

    written = 0
    try:
        archive = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile):
        raise ProjectError('That file is not a readable .zip')

    with archive:
        members = archive.infolist()
        if len(members) > MAX_MEMBERS:
            raise ProjectError(f'That zip holds {len(members)} entries, which '
                               'is more than a model export should.')
        for member in members:
            if member.is_dir():
                continue
            name = member.filename.replace('\\', '/')
            parts = [p for p in Path(name).parts
                     if p not in ('', '.', '..') and ':' not in p]
            if not parts:
                continue
            # Flattened: a zip made by right-clicking a folder holds one folder
            # with everything inside it, and the runner looks for labels.txt
            # beside the model rather than one level up from it.
            target = folder / parts[-1]
            with archive.open(member) as source, open(target, 'wb') as out:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_UNPACKED_BYTES:
                        raise ProjectError('That zip unpacks to more than this '
                                           'will accept.')
                    out.write(chunk)

    models = sorted(p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES)
    if not models:
        kinds = ', '.join(sorted(ALLOWED_SUFFIXES))
        raise ProjectError(
            f'No model file in that zip. Expected one of: {kinds}. A Custom '
            'Vision export folder has model.onnx in it.')
    return models[0]


def _slug(text):
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '-', str(text or '')).strip('-.')
    return (cleaned or 'model')[:60]


def root():
    IMPORTED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return IMPORTED_MODELS_DIR


def _unique_folder(name):
    base = root() / _slug(name)
    if not base.exists():
        return base
    for suffix in range(2, 100):
        candidate = root() / f'{_slug(name)}-{suffix}'
        if not candidate.exists():
            return candidate
    raise ProjectError('Too many imports under that name; rename it.')


def add(model_file, name=None, labels_file=None, conventions=None):
    """
    Store an uploaded model, with whatever came with it.

    Returns the same shape list_models produces, so the picker can show it
    beside the models this installation trained.
    """
    if model_file is None or not getattr(model_file, 'filename', ''):
        raise ProjectError('Choose a model file to import')

    filename = Path(model_file.filename).name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES and suffix != '.zip':
        raise ProjectError(
            f'Importable model files: {", ".join(sorted(ALLOWED_SUFFIXES))}, '
            'or a .zip of an export folder')

    folder = _unique_folder(name or Path(filename).stem)
    folder.mkdir(parents=True)
    try:
        if suffix == '.zip':
            # The whole export folder, so the labels and the preprocessing it
            # recorded travel with the model instead of being retyped.
            staged = folder / 'upload.zip'
            model_file.save(str(staged))
            target = _extract(staged, folder)
            staged.unlink(missing_ok=True)
        else:
            target = folder / f'model{suffix}'
            model_file.save(str(target))
        if not target.is_file() or target.stat().st_size == 0:
            raise ProjectError('That model file arrived empty')

        labels = []
        beside = folder / 'labels.txt'
        if beside.is_file():
            names = [line.strip() for line in
                     beside.read_text(encoding='utf-8-sig').splitlines()]
            labels = [n for n in names if n]
        if labels_file is not None and getattr(labels_file, 'filename', ''):
            raw = labels_file.read(MAX_LABEL_BYTES)
            text = raw.decode('utf-8-sig', errors='replace')
            labels = [line.strip() for line in text.replace(',', '\n').splitlines()
                      if line.strip()]
            if labels:
                (folder / 'labels.txt').write_text('\n'.join(labels) + '\n',
                                                  encoding='utf-8')

        chosen = ''
        if conventions:
            from services import onnxrunner
            parsed = onnxrunner.parse_conventions(conventions)
            if parsed:
                chosen = ' '.join(parsed[key] for key in
                                  ('resize', 'channels', 'scale', 'box_order')
                                  if key in parsed)
                (folder / 'conventions.txt').write_text(chosen + '\n',
                                                        encoding='utf-8')

        # What the file turns out to be, read from the file rather than taken
        # on trust: an import that cannot be opened should say so now, not on
        # the first labelling pass.
        detail = describe(target)

        (folder / 'import.json').write_text(json.dumps({
            'display_name': name or Path(filename).stem,
            'original_filename': filename,
            'imported_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'labels': labels,
            'conventions': chosen,
            'detail': detail,
        }, indent=2), encoding='utf-8')
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    return entry(folder)


def describe(model_path):
    """
    A one-line account of what was imported, or why it cannot be used.

    Opening the file at import time turns "the auto-labeller found nothing"
    into "this file could not be opened", said at the moment somebody can do
    something about it.
    """
    suffix = Path(model_path).suffix.lower()
    if suffix != '.onnx':
        return {'kind': suffix.lstrip('.'), 'readable': True}

    from services import onnxrunner
    try:
        detector = onnxrunner.OnnxDetector(model_path,
                                           display_name=Path(model_path).name)
    except ProjectError as exc:
        raise ProjectError(f'That model was not imported: {exc}')

    return {
        'kind': 'onnx',
        'readable': True,
        'source': detector.source,
        'input_size': detector.size,
        'feeding': (f'{detector.resize} {detector.channels} '
                    f'{detector.scale} {detector.box_order}'),
        'notes': list(getattr(detector, 'notes', [])),
    }


def entry(folder):
    """One imported model, in the shape the model pickers already expect."""
    folder = Path(folder)
    models = [p for p in folder.iterdir()
              if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES]
    if not models:
        return None
    model = models[0]

    meta = {}
    info = folder / 'import.json'
    if info.is_file():
        try:
            meta = json.loads(info.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            meta = {}

    display = meta.get('display_name') or folder.name
    return {
        'project': 'imported',
        'label': f'{display} (imported)',
        'model_name': display,
        'path': str(model),
        'checkpoint': 'best',      # so the pickers that filter on this show it
        'format': model.suffix.lstrip('.'),
        'size_mb': round(model.stat().st_size / (1024 * 1024), 2),
        'modified': model.stat().st_mtime,
        'imported': True,
        'labels': meta.get('labels') or [],
        'conventions': meta.get('conventions') or '',
        'detail': meta.get('detail') or {},
    }


def list_models():
    if not IMPORTED_MODELS_DIR.exists():
        return []
    found = []
    for folder in sorted(IMPORTED_MODELS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        try:
            item = entry(folder)
        except OSError:
            continue
        if item:
            found.append(item)
    return sorted(found, key=lambda m: m['modified'], reverse=True)


def set_labels(folder_name, labels_file):
    """
    Attach the class names to a model already imported.

    The two files are chosen separately in a browser, and refusing the model
    until its labels.txt is also in hand would mean the model cannot be tried
    at all by somebody who has mislaid it. Numbered classes are worth
    something; nothing is worth nothing.
    """
    folder = (root() / _slug(folder_name)).resolve()
    if folder.parent != root().resolve() or not folder.is_dir():
        raise ProjectError('No such imported model', status=404)
    if labels_file is None or not getattr(labels_file, 'filename', ''):
        raise ProjectError('Choose a labels file')

    raw = labels_file.read(MAX_LABEL_BYTES)
    text = raw.decode('utf-8-sig', errors='replace')
    labels = [line.strip() for line in text.replace(',', '\n').splitlines()
              if line.strip()]
    if not labels:
        raise ProjectError('That file held no class names')

    (folder / 'labels.txt').write_text('\n'.join(labels) + '\n', encoding='utf-8')

    info = folder / 'import.json'
    meta = {}
    if info.is_file():
        try:
            meta = json.loads(info.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            meta = {}
    meta['labels'] = labels
    info.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return entry(folder)


def remove(folder_name):
    """Delete one import. Only ever a folder directly under the import root."""
    folder = (root() / _slug(folder_name)).resolve()
    if folder.parent != root().resolve() or not folder.is_dir():
        raise ProjectError('No such imported model', status=404)
    shutil.rmtree(folder, ignore_errors=True)
    return {'removed': folder.name}


def conventions_for(model_path):
    """What the folder beside a model says about how to feed it."""
    note = Path(model_path).parent / 'conventions.txt'
    if note.is_file():
        try:
            return note.read_text(encoding='utf-8-sig').strip()
        except OSError:
            return ''
    return ''
