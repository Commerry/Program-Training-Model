"""
A finished run has to produce weights that detect something.

This exists because of a specific failure, and the failure was silent. Turning
on the dataloader's in-memory cache gave two runs the same reported mAP50 of
0.995; one of them then scored 0.01 on every held-out image. Every number the
application had said the run was excellent. Nothing short of loading the
weights and looking would have caught it.

So the run now loads its own weights and looks, and this checks that it does,
in both directions: that a model which detects is reported as usable, and that
one which detects nothing is reported as not, however good its metrics were.

Deliberately kept fast enough to run every time rather than only under --full,
because a check that catches this is worth nothing if nobody waits for it.

    python backend/tests/test_model_is_usable.py
"""
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_usable_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
from app import create_app                              # noqa: E402
from training.worker_common import self_check           # noqa: E402

app = create_app()
c = app.test_client()

fails = []
started = time.time()


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


def frame(x, y, seed):
    """A large bright block on a dark field: about as easy as detection gets."""
    rng = np.random.default_rng(seed)
    img = np.full((256, 256, 3), 30, np.uint8)
    img = cv2.add(img, rng.integers(0, 18, (256, 256, 3), dtype=np.uint8))
    cv2.rectangle(img, (x, y), (x + 90, y + 90), (240, 246, 252), -1)
    cv2.rectangle(img, (x + 16, y + 16), (x + 74, y + 74), (24, 30, 40), 4)
    return img


print('== the check, on weights that were never trained for this ==')
# The stock model has never seen these shapes, so it stands in for any run that
# produces weights which do not work, without needing to break a run to get one.
stub = TMP / 'stub'
(stub / 'images' / 'val').mkdir(parents=True)
for i in range(4):
    cv2.imwrite(str(stub / 'images' / 'val' / f'{i}.jpg'), frame(80, 80, i))

report = self_check(REPO / 'data' / 'weights' / 'yolo11n.pt', stub, 320,
                    lambda m: None)
check('the check ran', report is not None, report)
if report:
    check('it reports the weights as unusable', report['usable'] is False, report)
    check('and says how many it looked at', report['images_checked'] == 4, report)

print('\n== a missing dataset does not break the check ==')
check('an absent split returns nothing rather than raising',
      self_check(REPO / 'data' / 'weights' / 'yolo11n.pt', TMP / 'nope', 320,
                 lambda m: None) is None)

print('\n== a run that works reports itself as usable ==')
c.post('/api/projects', json={'name': 'usable'})
rng = np.random.default_rng(9)
uploads, boxes = [], []
for i in range(24):
    x, y = int(rng.integers(20, 140)), int(rng.integers(20, 140))
    uploads.append((io.BytesIO(cv2.imencode('.jpg', frame(x, y, i))[1].tobytes()),
                    f'{i:02d}.jpg'))
    boxes.append((x - 2, y - 2, 94, 94))

r = c.post('/api/projects/usable/images', data={'images': uploads},
           content_type='multipart/form-data')
names = [x['filename'] for x in r.get_json()['imported']]
for name, (x, y, w, h) in zip(names, boxes):
    c.post(f'/api/projects/usable/images/{name}/annotations',
           json={'regions': [{'tag': 'block', 'x': x, 'y': y,
                              'width': w, 'height': h}]})

print('    training (about a minute)...')
c.post('/api/projects/usable/training/start', json={
    'model_type': 'yolo11n', 'epochs': 40, 'batch_size': 8, 'img_size': 224,
    'export_formats': ['pt'], 'model_name': 'usable_run'})

deadline = time.time() + 900
status = {}
while time.time() < deadline:
    time.sleep(4)
    status = c.get('/api/projects/usable/training/status').get_json()['status'] or {}
    if status.get('status') in ('completed', 'failed', 'stopped', 'error'):
        break

check('the run completed', status.get('status') == 'completed', status.get('error'))
mAP = (status.get('metrics') or {}).get('mAP50')
print(f"    mAP50={mAP}")

result = status.get('self_check')
check('the run carries a self-check', isinstance(result, dict), result)
if isinstance(result, dict):
    print(f"    self-check: {result['images_with_detections']}/"
          f"{result['images_checked']} images, {result['detections']} objects, "
          f"best {result['best_score']}")
    check('it looked at some validation images', result['images_checked'] > 0, result)
    check('the trained model detects on its own validation images',
          result['usable'] is True,
          'the run reported a good mAP and then found nothing, which is the '
          'contradiction this test exists to catch')
    check('it records the threshold it used', result.get('threshold'), result)

print('\n== the metrics and the check must agree ==')
# This is the assertion that would have caught the caching failure: a run
# cannot report a good mAP and detect nothing. Either both are bad or both are
# good; one of each means the metric is measuring something the weights do not
# do.
if isinstance(result, dict) and mAP is not None:
    contradiction = mAP > 0.5 and not result['usable']
    check('a good mAP is not paired with a model that finds nothing',
          not contradiction,
          f'mAP50={mAP} but nothing detected on its own validation images')

print('\n== the history keeps it too ==')
history = c.get('/api/projects/usable/training/history').get_json()['history']
check('the finished run is in the history', bool(history), history)
if history:
    check('and carries its self-check', isinstance(history[0].get('self_check'), dict),
          sorted(history[0]))

print(f'\n({time.time() - started:.0f}s)')
print('MODEL IS USABLE OK' if not fails else f'{len(fails)} FAILED: {fails}')
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
