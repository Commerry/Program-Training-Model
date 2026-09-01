"""
A detector from somewhere else, pre-labelling a project that has nothing.

Auto-labelling saves the most work on a project with no annotations at all,
which is exactly the moment this installation has no model to offer. Somebody
arriving with a detector from a previous system should be able to point the
auto-labeller at it and find out in one pass whether correcting its boxes
beats drawing them.

An ONNX on its own cannot do that. It carries no class names, and nothing in
it says how it wants to be fed. So an import is a folder, and what is checked
here is that the folder travels with the model: that the boxes land in the
right place, and that they are labelled with the imported model's own class
names rather than whatever the project happens to call its tags -- which would
be wrong on every box while looking entirely correct.

    python backend/tests/test_imported_models.py
"""
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_import_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402
import torch.nn as nn                                   # noqa: E402

import config                                           # noqa: E402
config.IMPORTED_MODELS_DIR = TMP / 'imported'

from app import create_app                              # noqa: E402

app = create_app()
c = app.test_client()

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


SIZE = 320
BOX = [0.25, 0.25, 0.55, 0.75]        # normalised, as such exports report
CLASS = 1
SCORE = 0.87


class CustomVisionShaped(nn.Module):
    """
    The three outputs Azure Custom Vision exports a detector as.

    Fixed answers: what is under test is the plumbing around the model, not
    the model. The names are what make it recognisable, and recognition is
    what decides how the frame is prepared.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 2, 3, 2, 1)
        self.register_buffer('boxes', torch.tensor([[BOX]], dtype=torch.float32))
        self.register_buffer('classes', torch.tensor([[CLASS]], dtype=torch.int64))
        self.register_buffer('scores', torch.tensor([[SCORE]], dtype=torch.float32))

    def forward(self, x):
        nothing = self.stem(x).sum() * 0.0
        return (self.boxes + nothing,
                self.classes + nothing.to(torch.int64),
                self.scores + nothing)


model_path = TMP / 'model.onnx'
torch.onnx.export(CustomVisionShaped(), torch.zeros(1, 3, SIZE, SIZE),
                  str(model_path), opset_version=12, input_names=['image_tensor'],
                  output_names=['detected_boxes', 'detected_classes',
                                'detected_scores'], dynamo=False)

LABELS = 'Bad\nCuffTear\nGood\n'

print('== importing it ==')
r = c.post('/api/models/import', data={
    'model': (io.BytesIO(model_path.read_bytes()), 'model.onnx'),
    'labels_file': (io.BytesIO(LABELS.encode()), 'labels.txt'),
    'name': 'auto stack iteration 42',
}, content_type='multipart/form-data')
body = r.get_json() or {}
check('the import succeeds', r.status_code == 200, (r.status_code, body))

record = body if r.status_code == 200 else {}
check('it is recognised as a Custom Vision export',
      (record.get('detail') or {}).get('source') == 'Azure Custom Vision',
      record.get('detail'))
check('and the way it must be fed is recorded',
      (record.get('detail') or {}).get('feeding') == 'stretch bgr raw xyxy',
      (record.get('detail') or {}).get('feeding'))
check('the class names came along', record.get('labels') == ['Bad', 'CuffTear', 'Good'],
      record.get('labels'))

imported_path = record.get('path') or ''
check('it is stored where the loaders can reach it', bool(imported_path),
      imported_path)

print('\n== it shows up beside the models this server trained ==')
r = c.get('/api/models/imported')
listed = (r.get_json() or {}).get('models') or []
check('the listing has it', len(listed) == 1, len(listed))
check('and it is marked as imported',
      bool(listed and listed[0].get('imported')), listed[:1])

print('\n== a project with nothing in it ==')
r = c.post('/api/projects', json={'name': 'gloves'})
check('the project is created', r.status_code in (200, 201), r.status_code)

# A frame whose object sits where the model says it does, so a box in the
# wrong place is visible as one. 640x480 squashed into 320 scales x by 0.5
# and y by 2/3, so the normalised box above covers 160..352 across and
# 120..360 down in the photograph.
frame = np.full((480, 640, 3), 30, np.uint8)
cv2.rectangle(frame, (160, 120), (352, 360), (220, 225, 235), -1)
encoded = cv2.imencode('.jpg', frame)[1].tobytes()

r = c.post('/api/projects/gloves/images',
           data={'images': [(io.BytesIO(encoded), 'a.jpg'),
                            (io.BytesIO(encoded), 'b.jpg')]},
           content_type='multipart/form-data')
check('the photographs upload', r.status_code == 200, r.status_code)

print('\n== previewing the imported model on it ==')
r = c.post('/api/projects/gloves/auto-label/preview',
           json={'model_path': imported_path, 'score_threshold': 0.5})
body = r.get_json() or {}
check('the preview runs', r.status_code == 200, (r.status_code, r.get_json()))
check('and it finds the object',
      bool(body.get('usable')), body.get('verdict'))
check('naming it from the imported labels, not the project tags',
      (body.get('results') or [{}])[0].get('tags') == ['CuffTear'],
      (body.get('results') or [{}])[0].get('tags'))

print('\n== and labelling with it ==')
r = c.post('/api/projects/gloves/auto-label',
           json={'model_path': imported_path, 'score_threshold': 0.5})
check('the job starts', r.status_code == 200, (r.status_code, r.get_json()))

import time                                             # noqa: E402
for _ in range(120):
    status = ((c.get('/api/projects/gloves/auto-label').get_json() or {})
              .get('job') or {})
    if status.get('status') in ('finished', 'completed', 'failed', 'cancelled'):
        break
    time.sleep(0.5)
check('it finishes', status.get('status') in ('finished', 'completed'), status)

r = c.get('/api/projects/gloves/images')
images = (r.get_json() or {}).get('images') or []
labelled = [i for i in images if i.get('annotated')]
check('both photographs came back labelled', len(labelled) == 2,
      [(i['filename'], i.get('regions_count')) for i in images])

# Read from disk: the annotations are written as ordinary files and there is
# no GET endpoint for one image's boxes -- the editor gets them with the image.
# An upload is stored under a timestamped name, so a.jpg is looked up by the
# original_name it was filed under rather than assumed.
with app.app_context():
    from services import projects as projects_service
    stored_name = next((i['filename'] for i in images
                        if i.get('original_name') == 'a.jpg'),
                       images[0]['filename'] if images else 'a.jpg')
    stored = projects_service.read_annotation('gloves', stored_name) or {}
regions = stored.get('regions') or []
check('one box per photograph', len(regions) == 1, regions)
if regions:
    box = regions[0]
    check('tagged with the imported class name', box.get('tag') == 'CuffTear',
          box.get('tag'))
    corners = [box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']]
    check('and drawn where the object is',
          all(abs(a - b) <= 6 for a, b in zip(corners, [160, 120, 352, 360])),
          [round(v) for v in corners])
    # The mark is on the image, not on each box: what a person needs to know
    # is that this picture was pre-labelled and wants checking.
    check('the picture is marked as pre-labelled', bool(stored.get('auto_labelled')),
          {k: stored.get(k) for k in ('auto_labelled', 'auto_label')})
    check('and it records which model did it',
          Path((stored.get('auto_label') or {}).get('model', '')).name.startswith('model'),
          stored.get('auto_label'))

print('\n== a file that is not a model ==')
r = c.post('/api/models/import', data={
    'model': (io.BytesIO(b'not a model at all'), 'model.onnx'),
}, content_type='multipart/form-data')
check('it is refused at import, not on the first labelling pass',
      r.status_code == 400, r.status_code)
check('and nothing is left behind',
      len(((c.get('/api/models/imported').get_json() or {}).get('models')) or []) == 1)

print('\n== removing one ==')
folder = Path(imported_path).parent.name
r = c.delete(f'/api/models/imported/{folder}')
check('it is removed', r.status_code == 200, r.status_code)
remaining = ((c.get('/api/models/imported').get_json() or {}).get('models')) or []
check('and the listing is empty again', remaining == [], remaining)

print('\n' + ('IMPORTED MODELS OK' if not fails else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
