"""
The path somebody actually walks through the studio, end to end.

Every piece of this was tested on its own and the walk through them was not,
which is how it came to be handed over not working. So this is the sequence as
a person does it, through the same endpoints the page calls, in the same
order, with the files they actually have:

    a zip of a Custom Vision dataset export
      -> a project made for it, imported, and offered as a source
      -> defects collected from it
    good gloves of another colour uploaded
      -> defects put on them, boxed
      -> the result exported as a dataset that can be trained on

And the two mistakes that are easy to make, because the answer to both has to
be a sentence somebody can act on rather than a silence:

    a model zip where a dataset belongs
    a dataset zip where a model belongs

    python backend/tests/test_studio_flow.py
"""
import io
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_studio_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402

import config                                           # noqa: E402
from app import create_app                              # noqa: E402
from services import defectlib, projects                # noqa: E402

config.REPO_ROOT = TMP
config.IMPORTED_MODELS_DIR = TMP / 'imported'
defectlib.DEFECT_LIBRARY_DIR = TMP / 'defect-library'

app = create_app()
c = app.test_client()

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


BG, PALE, DARK = 232, 176, 70
CLASSES = ['DirtM', 'CuffTear', 'Hole', 'Good']


def frame(level, seed, defect=None):
    """A backlit glove of one grey, with a real defect in it when asked."""
    rng = np.random.default_rng(seed)
    img = np.full((320, 320), float(BG), np.float32)
    cv2.ellipse(img, (160, 170), (95, 120), 0, 0, 360, float(level), -1)
    if defect:
        kind, (cx, cy), radius = defect
        patch = np.zeros_like(img)
        cv2.circle(patch, (cx, cy), radius, 1.0, -1)
        patch = cv2.GaussianBlur(patch, (9, 9), 2.0)
        img = (img * (1 - patch) + BG * patch if kind == 'hole'
               else img * np.exp(-patch * 0.6))
    img += rng.standard_normal(img.shape).astype(np.float32) * 1.3
    return np.clip(img, 0, 255).astype(np.uint8)


print('== the dataset Custom Vision exports, zipped ==')
# images/ beside labels/ and a label.txt, nested in one folder the way
# right-clicking it zips it. The class order is deliberately not alphabetical.
bundle = TMP / 'dataset.zip'
with zipfile.ZipFile(bundle, 'w') as archive:
    archive.writestr('export/label.txt', '\n'.join(CLASSES) + '\n')
    for i in range(8):
        kind = 'hole' if i % 2 else 'stain'
        spot = (140 + i * 5, 160 + i * 4)
        png = cv2.imencode('.png', frame(PALE, 10 + i, (kind, spot, 14)))[1]
        archive.writestr(f'export/images/g{i}.png', png.tobytes())
        index = CLASSES.index('Hole' if kind == 'hole' else 'DirtM')
        good = CLASSES.index('Good')
        # Two lines per picture: the glove, and the defect on it. A defect is
        # stored as a fraction of its glove, so a dataset without the glove
        # boxed has nothing to take the fraction of.
        archive.writestr(
            f'export/labels/g{i}.txt',
            f'{good} 0.5 0.53 0.60 0.75\n'
            f'{index} {spot[0] / 320:.6f} {spot[1] / 320:.6f} '
            f'{36 / 320:.6f} {36 / 320:.6f}')

print('    a zip of 8 images, 8 label files and a label.txt')

# The studio makes the project itself, then posts the zip at it.
r = c.post('/api/projects', json={'name': 'defects-from-zip'})
check('the project is created', r.status_code in (200, 201), r.get_json())

r = c.post('/api/projects/defects-from-zip/import-dataset',
           data={'file': (bundle.open('rb'), 'dataset.zip')},
           content_type='multipart/form-data')
body = r.get_json() or {}
check('the dataset zip is accepted', r.status_code == 200,
      (r.status_code, body.get('message')))
check('and read as YOLO', (body.get('job') or {}).get('format') == 'yolo',
      body.get('job'))

with app.app_context():
    from services import datasetimport
    status = datasetimport.wait_for_idle('defects-from-zip', timeout=180)
print(f"    {status.get('message')}")
check('every picture came in', status.get('imported') == 8, status)
check('with its glove and its defect', status.get('boxes') == 16, status)

listed = (c.get('/api/projects/defects-from-zip/images').get_json() or {}).get('images') or []
tags = sorted({t for i in listed for t in (i.get('tags') or [])})
check('named from label.txt, in its own order',
      tags == ['DirtM', 'Good', 'Hole'], tags)

print('\n== collecting the defects from it ==')
r = c.post('/api/projects/defects-from-zip/defect-library',
           json={'labels': ['DirtM', 'Hole'], 'per_label': 20})
body = r.get_json() or {}
check('the collection runs', r.status_code == 200, body)
# Sixteen boxes went in; eight of them are gloves, which are the reference
# rather than something to collect.
check('it found every defect, and no gloves', body.get('total') == 8,
      body.get('total'))
per_tag = {row['tag']: row for row in (body.get('per_tag') or [])}
check('a hole is light coming through',
      per_tag.get('Hole', {}).get('defect_class') == 'hole', per_tag)
check('and dirt is something on the rubber',
      per_tag.get('DirtM', {}).get('defect_class') == 'stain', per_tag)

print('\n== the good gloves of the new colour ==')
# The page creates the project on the first drop, so this is that request.
r = c.post('/api/projects', json={'name': 'defect-studio-today'})
check('somewhere to keep them', r.status_code in (200, 201), r.get_json())

files = [(io.BytesIO(cv2.imencode('.png', frame(DARK, 100 + i))[1].tobytes()),
          f'good{i}.png') for i in range(6)]
r = c.post('/api/projects/defect-studio-today/images',
           data={'images': files}, content_type='multipart/form-data')
check('they upload', r.status_code == 200, r.get_json())

# Where the glove is, without marking these as labelled pictures: they are
# still the good gloves waiting to have something put on them.
with app.app_context():
    for item in (r.get_json() or {}).get('imported') or []:
        record = projects.read_annotation('defect-studio-today',
                                          item['filename']) or {}
        record['regions'] = [{'tag': 'Good', 'x': 64, 'y': 50,
                              'width': 192, 'height': 240}]
        record['annotated'] = False
        projects.write_annotation('defect-studio-today', item['filename'], record)
    projects.rebuild_index('defect-studio-today')

listed = (c.get('/api/projects/defect-studio-today/images').get_json() or {}).get('images') or []
check('six of them, none labelled',
      len(listed) == 6 and not any(i.get('annotated') for i in listed),
      len(listed))

print('\n== putting one on the other ==')
r = c.post('/api/projects/defect-studio-today/defect-synth',
           json={'source_project': 'defects-from-zip',
                 'tags': ['DirtM', 'Hole'], 'per_image': 1})
check('the run starts', r.status_code == 200, r.get_json())

with app.app_context():
    status = defectlib.wait_for_idle('defect-studio-today', timeout=240)
print(f"    {status.get('message')}")
check('it finishes', status.get('status') == 'finished', status)
check('and made something', (status.get('made') or 0) >= 3, status.get('made'))

after = (c.get('/api/projects/defect-studio-today/images').get_json() or {}).get('images') or []
made = [i for i in after if i.get('annotated')]
check('the made ones are boxed', len(made) == status.get('made'),
      (len(made), status.get('made')))
check('the good gloves are untouched',
      len([i for i in after if not i.get('annotated')]) == 6,
      len([i for i in after if not i.get('annotated')]))

print('\n== what the page needs to draw them ==')
sample = made[0] if made else {}
check('each carries its boxes', bool(sample.get('boxes')), sample.get('boxes'))
check('and its size, to place them', bool(sample.get('width')), sample.get('width'))
check('and the good glove it was made from, for the comparison',
      bool(sample.get('original_name')), sample.get('original_name'))
check('marked as made rather than photographed',
      bool(sample.get('synthetic')), sample.get('synthetic'))

print('\n== and out again, as a dataset ==')
r = c.post('/api/projects/defect-studio-today/export', json={})
check('the export runs', r.status_code == 200, r.status_code)
check('it is a zip', r.get_data()[:2] == b'PK', r.get_data()[:4])
with zipfile.ZipFile(io.BytesIO(r.get_data())) as archive:
    names = archive.namelist()
check('with the pictures in it',
      sum(1 for n in names if n.startswith('images/')) >= len(made),
      len([n for n in names if n.startswith('images/')]))
check('and their boxes', 'dataset.json' in names, names[:5])

print('\n== training can still be prepared from it ==')
# Not a full run -- that is the slow suite -- but the step that turns a
# project into something a trainer reads, which is what breaks first when
# annotations change shape.
r = c.post('/api/projects/defect-studio-today/prepare-dataset')
body = r.get_json() or {}
check('the dataset builds', r.status_code == 200, (r.status_code, body))
built = body.get('dataset') or {}
check('with the made images in the training set',
      (built.get('train_images') or 0) > 0, built)
# Which defect classes turn up depends on which the run happened to draw,
# and Good is in there because the good gloves carry their own box -- an
# image with a box on it is labelled data whether or not a defect was added.
check('and at least one synthesised defect class is in it',
      bool({'DirtM', 'Hole'} & set(built.get('classes') or [])),
      built.get('classes'))

print('\n== the two easy mistakes ==')
# A model zip where a dataset belongs.
model_zip = TMP / 'model.zip'
with zipfile.ZipFile(model_zip, 'w') as archive:
    archive.writestr('Iteration42/model.onnx', b'weights, not pictures')
    archive.writestr('Iteration42/labels.txt', 'Hole\nDirtM\n')

r = c.post('/api/projects', json={'name': 'wrong-way-round'})
r = c.post('/api/projects/wrong-way-round/import-dataset',
           data={'file': (model_zip.open('rb'), 'model.zip')},
           content_type='multipart/form-data')
message = (r.get_json() or {}).get('message', '')
check('a model zip is refused as a dataset', r.status_code == 400,
      (r.status_code, message[:90]))
check('and the message says what was missing',
      'image' in message.lower(), message[:160])

# A dataset zip where a model belongs.
r = c.post('/api/models/import',
           data={'model': (bundle.open('rb'), 'dataset.zip')},
           content_type='multipart/form-data')
message = (r.get_json() or {}).get('message', '')
check('a dataset zip is refused as a model', r.status_code == 400,
      (r.status_code, message[:90]))
check('and that message says what was expected',
      'model' in message.lower(), message[:160])

print('\n== nothing was lost along the way ==')
with app.app_context():
    for name, expected in (('defects-from-zip', 8),
                           ('defect-studio-today', 6 + len(made))):
        stored = len(list((projects.images_dir(name)).iterdir()))
        check(f'{name} still has every file on disk', stored == expected,
              (stored, expected))
        meta = projects.get_project(name)
        check(f'{name} counts what is there',
              meta['total_images'] == expected, (meta['total_images'], expected))

print('\n' + ('STUDIO FLOW OK' if not fails else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
