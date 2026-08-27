"""
Train a model for real, then use it.

Every other test stops at "the dataset was built" or "the process started".
The original complaint about this application was that training took hours and
produced a model that could not detect anything, and neither half of that shows
up until a run goes all the way through: dataset -> subprocess -> weights ->
inference on an image the model never saw.

The task is deliberately easy — a white square against a blue circle — and the
model is the smallest YOLO at 192px, so the run finishes on a CPU in a few
minutes. It still has to train long enough to actually detect something: a
two-epoch run completes and writes weights that find nothing at any threshold,
which would let a broken pipeline pass. Reaching a real detection is the point.

    python backend/tests/test_train_end_to_end.py
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
TMP = Path(tempfile.mkdtemp(prefix='vt_e2e_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
# Reuse the checkout's pretrained weights so the run does not need the network.
os.environ.setdefault('WEIGHTS_ROOT', str(REPO / 'data' / 'weights'))
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
from app import create_app                              # noqa: E402

app = create_app()
c = app.test_client()

fails = []
started = time.time()


def check(label, cond, detail=''):
    mark = 'PASS ' if cond else 'FAIL '
    print(f'  {mark}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


def draw(width, height, shape, position, rng):
    """A frame with one object on it, and the box that contains it."""
    img = np.full((height, width, 3), 38, np.uint8)
    img = cv2.add(img, rng.integers(0, 22, (height, width, 3), dtype=np.uint8))
    x, y = position
    size = 76
    if shape == 'square':
        cv2.rectangle(img, (x, y), (x + size, y + size), (232, 240, 250), -1)
    else:
        cv2.circle(img, (x + size // 2, y + size // 2), size // 2,
                   (90, 200, 250), -1)
    return img, (x - 3, y - 3, size + 6, size + 6)


PROJECT = 'e2e'
c.post('/api/projects', json={'name': PROJECT})

print('== build a small but real dataset ==')
rng = np.random.default_rng(11)
uploads, boxes = [], []
for i in range(48):
    shape = 'square' if i % 2 == 0 else 'circle'
    position = (int(rng.integers(40, 300)), int(rng.integers(40, 220)))
    img, box = draw(416, 352, shape, position, rng)
    uploads.append((io.BytesIO(cv2.imencode('.jpg', img)[1].tobytes()),
                    f'{shape}_{i:02d}.jpg'))
    boxes.append((shape, box))

r = c.post(f'/api/projects/{PROJECT}/images', data={'images': uploads},
           content_type='multipart/form-data')
names = [x['filename'] for x in r.get_json()['imported']]
check('48 images uploaded', len(names) == 48, len(names))

for name, (shape, (x, y, w, h)) in zip(names, boxes):
    c.post(f'/api/projects/{PROJECT}/images/{name}/annotations',
           json={'regions': [{'tag': shape, 'x': x, 'y': y,
                              'width': w, 'height': h}]})

summary = c.get(f'/api/projects/{PROJECT}/dataset-summary').get_json()
check('every image is annotated', summary['annotated_images'] == 48,
      summary['annotated_images'])
check('both classes are present', len(summary.get('tags') or {}) == 2,
      summary.get('tags'))

print('\n== the labels the trainer will actually read ==')
report = c.post(f'/api/projects/{PROJECT}/prepare-dataset').get_json()['dataset']
dataset_root = Path(report['dataset_path'])
label_files = sorted((dataset_root / 'labels' / 'train').glob('*.txt'))
check('a label file per training image',
      len(label_files) == report['train_images'],
      (len(label_files), report['train_images']))

# The defect that made every earlier model useless: boxes normalised against a
# 1x1 image, so every label read as the full frame.
degenerate, out_of_range = [], []
for path in label_files:
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
            out_of_range.append((path.name, line))
        if bw > 0.95 and bh > 0.95:
            degenerate.append((path.name, line))
check('every label is inside [0,1]', not out_of_range, out_of_range[:2])
check('no label covers the whole frame', not degenerate, degenerate[:2])

print('\n== options the UI offers are the ones the trainer accepts ==')
options = c.get(f'/api/projects/{PROJECT}/training/options').get_json()
model_ids = [m['id'] if isinstance(m, dict) else m
             for m in (options.get('models') or options.get('model_types') or [])]
check('a model list is offered', bool(model_ids), options.keys())
check('the smallest YOLO is among them', any('n' in m for m in model_ids), model_ids)

print('\n== start a real run ==')
r = c.post(f'/api/projects/{PROJECT}/training/start', json={
    'model_type': 'yolo11n',
    'epochs': 60,
    'batch_size': 8,
    'img_size': 192,
    'export_formats': ['pt'],
    'model_name': 'e2e_run',
})
body = r.get_json()
check('the run was accepted', r.status_code == 200, body)

print('    waiting for it to finish (a few minutes on a CPU)...')
DEADLINE = time.time() + 900
state, last_progress = None, ''
while time.time() < DEADLINE:
    status = c.get(f'/api/projects/{PROJECT}/training/status').get_json()['status']
    state = (status or {}).get('status')
    progress = f"{state} {(status or {}).get('progress', '')}"
    if progress != last_progress:
        print(f'      {progress}')
        last_progress = progress
    if state in ('completed', 'failed', 'stopped', 'error'):
        break
    time.sleep(4)

check('the run reached completion', state == 'completed', state)

log_lines = c.get(f'/api/projects/{PROJECT}/training/logs').get_json()['logs']
check('the log was captured', len(log_lines) >= 5, len(log_lines))
check('the log records the work it did',
      any('epoch' in line.lower() or 'complete' in line.lower() for line in log_lines),
      log_lines[-3:])
if state != 'completed':
    print('\n--- last of the training log ---')
    print('\n'.join(log_lines[-25:]))

print('\n== the run produced weights ==')
models = c.get(f'/api/projects/{PROJECT}/training/models').get_json()
entries = models.get('models') or []
check('at least one model file is listed', bool(entries), models)

weight_path = None
if entries:
    first = entries[0]
    weight_path = first.get('path') if isinstance(first, dict) else first
    size = Path(weight_path).stat().st_size if weight_path and Path(weight_path).exists() else 0
    check('the weights file exists and is not empty', size > 100_000, size)

print('\n== the trained model detects the thing it was trained on ==')
if weight_path and Path(weight_path).exists():
    # An image the model has never seen, drawn the same way as the training set.
    probe_rng = np.random.default_rng(99)
    probe, probe_box = draw(416, 352, 'square', (150, 120), probe_rng)
    encoded = cv2.imencode('.jpg', probe)[1].tobytes()

    # By path: the model this server just trained, with no round trip.
    r = c.post('/api/models/test',
               data={'images': (io.BytesIO(encoded), 'probe.jpg'),
                     'model_path': str(weight_path),
                     'score_threshold': '0.05'},
               content_type='multipart/form-data')
    result = r.get_json()
    check('inference by server-side path returned a result',
          r.status_code == 200, result)
    per_image = (result or {}).get('results') or []
    detections = per_image[0].get('detections', []) if per_image else []
    check('the model detects the object it was trained on', bool(detections),
          'the run completed and wrote weights, but they find nothing — '
          'a pipeline that trains on wrong labels looks exactly like this')
    if detections:
        best = max(detections, key=lambda d: d.get('score', 0))
        print(f"      best: {best.get('label_name')} at {best.get('score', 0):.3f} "
              f"box {best.get('box')}")
        check('the detection carries a class name and a box',
              bool(best.get('label_name')) and len(best.get('box') or []) == 4, best)
        check('it named the class that was drawn',
              best.get('label_name') == 'square', best.get('label_name'))

        # The point of the whole test: a box in roughly the right place proves
        # the labels the model learned from described the actual pixels. The
        # defect this application shipped with normalised every box against a
        # 1x1 image, and a model trained on that puts its boxes anywhere.
        tx, ty, tw, th = probe_box
        bx0, by0, bx1, by1 = best['box']
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        inside = tx <= cx <= tx + tw and ty <= cy <= ty + th
        check('the box sits on the object rather than somewhere else', inside,
              f'centre ({cx:.0f},{cy:.0f}) is outside {probe_box}')
    check('the class names came back with it',
          bool(result.get('resolved_label_names')), result.get('resolved_label_names'))

    # By upload: the path every model from elsewhere takes into this screen.
    r = c.post('/api/models/test',
               data={'images': (io.BytesIO(encoded), 'probe.jpg'),
                     'model': (io.BytesIO(Path(weight_path).read_bytes()), 'best.pt'),
                     'score_threshold': '0.05'},
               content_type='multipart/form-data')
    check('inference by upload still works', r.status_code == 200, r.get_json())

    print('\n== the list the model picker reads ==')
    listed = c.get('/api/models').get_json()['models']
    check('the trained model is offered for testing',
          any(m['path'] == str(weight_path) for m in listed),
          [m.get('path') for m in listed][:3])
    check('each entry names its project and run',
          all(m.get('project') and m.get('run') for m in listed), listed[:1])

    print('\n== a path outside the projects tree is refused ==')
    r = c.post('/api/models/test',
               data={'images': (io.BytesIO(encoded), 'probe.jpg'),
                     'model_path': str(REPO / 'backend' / 'app.py')},
               content_type='multipart/form-data')
    check('a file outside the projects tree is rejected',
          r.status_code in (400, 403), r.status_code)
else:
    check('the trained model detects the thing it was trained on', False,
          'no weights to test')

print('\n== the run is recorded in history ==')
history = c.get(f'/api/projects/{PROJECT}/training/history').get_json()
runs = history.get('history') or history.get('runs') or []
check('the project history has the run', bool(runs), history)

overview = c.get('/api/overview').get_json()
check('the dashboard overview counts the project',
      overview.get('project_count', 0) >= 1, overview.get('project_count'))
check('the dashboard overview counts the finished run',
      overview.get('completed_runs', 0) >= 1, overview.get('completed_runs'))
check('the dashboard overview carries a mAP figure',
      'average_map50' in overview, sorted(overview))

global_history = c.get('/api/history').get_json()
check('the global history lists it',
      bool(global_history.get('history') or global_history.get('runs')),
      global_history)

print('\n== the status can be reset for another run ==')
c.post(f'/api/projects/{PROJECT}/training/reset')
after = c.get(f'/api/projects/{PROJECT}/training/status').get_json()
check('resetting clears the finished status',
      (after.get('status') or {}).get('status') in (None, 'idle', 'ready'),
      after)

elapsed = time.time() - started
print(f'\n({elapsed:.0f}s)')
print('END TO END OK' if not fails else f'{len(fails)} FAILED: {fails}')
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
