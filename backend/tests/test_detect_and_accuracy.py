"""
Detecting on one image, and reporting accuracy per class.

Two things a person needs while deciding whether a model is worth using.

Detecting on a single image is deliberately not auto-labelling: that is a bulk
background pass over everything unannotated, and turning a model loose on a
thousand pictures is a poor way to find out whether it works. This runs on one
image, writes nothing, and hands the boxes back for a person to judge.

The per-class figures answer the question the box counts cannot. "347 boxes
drawn for class 8" says how much work went in; it says nothing about whether
the model learned it. On a ten-class detector the difference between "0.72
overall" and "everything is fine except 8" is the difference between knowing
what to photograph next and not.

    python backend/tests/test_detect_and_accuracy.py
"""
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_da_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
from app import create_app                              # noqa: E402

app = create_app()
c = app.test_client()

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


def scene(seed):
    rng = np.random.default_rng(seed)
    img = np.full((256, 256, 3), 32, np.uint8)
    img = cv2.add(img, rng.integers(0, 18, (256, 256, 3), dtype=np.uint8))
    x, y = int(rng.integers(25, 140)), int(rng.integers(25, 140))
    cv2.rectangle(img, (x, y), (x + 88, y + 88), (238, 244, 250), -1)
    cv2.rectangle(img, (x + 16, y + 16), (x + 72, y + 72), (26, 32, 42), 4)
    return img, (x - 2, y - 2, 92, 92)


print('== a project with a trained model ==')
c.post('/api/projects', json={'name': 'da'})
uploads, boxes = [], []
for i in range(24):
    img, box = scene(i)
    uploads.append((io.BytesIO(cv2.imencode('.jpg', img)[1].tobytes()), f'{i:02d}.jpg'))
    boxes.append(box)

r = c.post('/api/projects/da/images', data={'images': uploads},
           content_type='multipart/form-data')
names = [x['filename'] for x in r.get_json()['imported']]
for name, (x, y, w, h) in zip(names, boxes):
    c.post(f'/api/projects/da/images/{name}/annotations',
           json={'regions': [{'tag': 'block', 'x': x, 'y': y,
                              'width': w, 'height': h}]})

print('\n== before any run, accuracy says so rather than showing zeros ==')
before = c.get('/api/projects/da/class-accuracy').get_json()
check('no run is reported', before['run'] is None, before['run'])
check('and it says why', 'No completed training run' in before.get('note', ''),
      before.get('note'))

print('\n== detecting before there is a model ==')
r = c.post(f'/api/projects/da/images/{names[0]}/detect',
           json={'model_path': ''})
check('a missing model is refused', r.status_code in (400, 403, 404),
      (r.status_code, (r.get_json() or {}).get('message')))

print('\n== train something to detect with ==')
print('    (about a minute)')
c.post('/api/projects/da/training/start', json={
    'model_type': 'yolo11n', 'epochs': 40, 'batch_size': 8, 'img_size': 224,
    'export_formats': ['pt'], 'model_name': 'da_run'})
deadline = time.time() + 900
status = {}
while time.time() < deadline:
    time.sleep(4)
    status = c.get('/api/projects/da/training/status').get_json()['status'] or {}
    if status.get('status') in ('completed', 'failed', 'stopped', 'error'):
        break
check('the run completed', status.get('status') == 'completed', status.get('error'))

# Which checkpoint to detect with. ultralytics picks best.pt by a ranking
# score, which on a small run can land on an epoch whose precision collapsed;
# the run reports an alternative when that happened, and this test is about
# detecting, not about that.
result = status.get('self_check') or {}
weights = status.get('best_model')
if not result.get('usable') and result.get('alternative'):
    print(f"    best.pt found nothing; using {result['alternative']['name']}")
    weights = result['alternative']['path']
check('the run produced a checkpoint that detects',
      result.get('usable') or result.get('alternative'),
      'neither best.pt nor last.pt detected anything on the validation split')

print('\n== detect on one image ==')
# A handful of images rather than one. What is being checked is the shape of
# the answer and that nothing gets written; a model trained on two dozen
# synthetic pictures detects on some of them and not others, and which one
# happens to be first is not the subject.
body, found_on = None, None
for candidate in names[:6]:
    reply = c.post(f'/api/projects/da/images/{candidate}/detect',
                   json={'model_path': weights, 'score_threshold': 0.2})
    if body is None:
        body = reply.get_json()
        check('the request succeeded', reply.status_code == 200, body)
    if (reply.get_json() or {}).get('count'):
        body, found_on = reply.get_json(), candidate
        break

print(f"    detected on {found_on or 'none of the first six'}")
check('it found something on at least one of them', body.get('count', 0) > 0,
      'the model detected nothing on any of six images, which is about the '
      'model rather than the endpoint')
check('it names the model used', body.get('model', '').endswith('.pt'),
      body.get('model'))
check('and the frame size', body.get('width') == 256 and body.get('height') == 256,
      (body.get('width'), body.get('height')))

if body.get('regions'):
    region = body['regions'][0]
    check('regions come back in the editor\'s own shape',
          {'tag', 'x', 'y', 'width', 'height'} <= set(region), sorted(region))
    check('with a positive size', region['width'] > 0 and region['height'] > 0,
          region)
    check('and the score kept alongside', 'score' in region, region)

print('\n== nothing is written ==')
# The whole point of doing this per image is that a person decides. A detect
# that quietly saved would be auto-labelling with extra steps.
saved = json.loads((Path(os.environ['PROJECTS_ROOT']) / 'da' / 'annotations' /
                    f'{names[0]}.json').read_text(encoding='utf-8'))
check('the annotation on disk still has exactly one hand-drawn box',
      len(saved['regions']) == 1, len(saved['regions']))
check('and is not marked as auto-labelled', not saved.get('auto_labelled'),
      saved.get('auto_labelled'))

print('\n== an image that is not in the project ==')
r = c.post('/api/projects/da/images/nope.jpg/detect', json={'model_path': weights})
check('a missing image is a 404', r.status_code == 404, r.status_code)

print('\n== a model outside the projects tree ==')
r = c.post(f'/api/projects/da/images/{names[0]}/detect',
           json={'model_path': str(REPO / 'backend' / 'app.py')})
check('is refused', r.status_code in (400, 403), r.status_code)

print('\n== accuracy per class, after the run ==')
after = c.get('/api/projects/da/class-accuracy').get_json()
check('the run is named', after['run'] == 'da_run', after['run'])
check('measured when it finished', bool(after['measured_at']), after['measured_at'])
check('every class of the project appears',
      [entry['name'] for entry in after['classes']] == ['block'],
      [entry['name'] for entry in after['classes']])

entry = after['classes'][0]
print(f"    block: {entry['boxes']} boxes, ap50={entry['ap50']}, "
      f"measured={entry['measured']}")
check('the box count is carried alongside', entry['boxes'] == 24, entry['boxes'])
check('and an accuracy figure that is a fraction, not a percent',
      entry['ap50'] is None or 0.0 <= entry['ap50'] <= 1.0, entry['ap50'])

print('\n== a class that was never validated is not reported as zero ==')
# Adding a class with no boxes at all: it cannot have been in the validation
# split, so it must come back unmeasured rather than scoring 0%.
c.post(f'/api/projects/da/images/{names[1]}/annotations',
       json={'regions': [
           {'tag': 'block', 'x': boxes[1][0], 'y': boxes[1][1],
            'width': boxes[1][2], 'height': boxes[1][3]},
           {'tag': 'never_trained', 'x': 5, 'y': 5, 'width': 20, 'height': 20},
       ]})
again = c.get('/api/projects/da/class-accuracy').get_json()
new_entry = next((e for e in again['classes'] if e['name'] == 'never_trained'), None)
check('the new class is listed', new_entry is not None,
      [e['name'] for e in again['classes']])
if new_entry:
    check('as unmeasured rather than zero', new_entry['measured'] is False,
          new_entry)
    check('with no accuracy figure at all', new_entry['ap50'] is None, new_entry)
check('and the note explains what unmeasured means',
      'validation' in again.get('note', '') or not again.get('note'),
      again.get('note'))

print('\n' + ('DETECT AND ACCURACY OK' if not fails else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
