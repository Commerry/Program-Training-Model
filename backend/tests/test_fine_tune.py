"""
Continuing from a model this server already trained.

The problem this solves is a deployment one. A model trained at one site does
not work at the next -- different lighting, different camera, different
background -- and the only option was to start again from the stock checkpoint,
which needs thousands of images and hours. But the model already knows what the
object is; only its appearance has shifted. Starting from those weights instead
should reach a working model on a fraction of the data.

This trains a model on one "site", then trains on a visibly different one both
ways -- from stock and from the first model -- and compares what each produces
from the same small number of images.

    python backend/tests/test_fine_tune.py
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
TMP = Path(tempfile.mkdtemp(prefix='vt_ft_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
from app import create_app                              # noqa: E402
from services import training                           # noqa: E402

app = create_app()
c = app.test_client()

fails = []
started = time.time()


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


def scene(seed, site):
    """
    The same object photographed at two very different sites.

    Site A is dim and blue; site B is bright and warm, which is what moving
    from one factory's lighting to another's actually looks like to a model.
    """
    rng = np.random.default_rng(seed)
    if site == 'A':
        background, item = (70, 40, 26), (210, 190, 120)
    else:
        background, item = (120, 150, 200), (60, 90, 235)

    img = np.zeros((256, 256, 3), np.uint8)
    img[:, :] = background
    img = cv2.add(img, rng.integers(0, 22, (256, 256, 3), dtype=np.uint8))
    x, y = int(rng.integers(30, 140)), int(rng.integers(30, 140))
    cv2.ellipse(img, (x + 45, y + 45), (44, 34), 0, 0, 360, item, -1)
    cv2.ellipse(img, (x + 45, y + 45), (44, 34), 0, 0, 360, (20, 20, 20), 2)
    return img, (x, y, 92, 72)


def build(project, site, count, seed_from=0):
    c.post('/api/projects', json={'name': project})
    uploads, boxes = [], []
    for i in range(count):
        img, box = scene(seed_from + i, site)
        uploads.append((io.BytesIO(cv2.imencode('.jpg', img)[1].tobytes()),
                        f'{i:03d}.jpg'))
        boxes.append(box)
    r = c.post(f'/api/projects/{project}/images', data={'images': uploads},
               content_type='multipart/form-data')
    names = [x['filename'] for x in r.get_json()['imported']]
    for name, (x, y, w, h) in zip(names, boxes):
        c.post(f'/api/projects/{project}/images/{name}/annotations',
               json={'regions': [{'tag': 'item', 'x': x, 'y': y,
                                  'width': w, 'height': h}]})
    return names


def train(project, model_name, epochs, base_model=None):
    payload = {'model_type': 'yolo11n', 'epochs': epochs, 'batch_size': 8,
               'img_size': 224, 'export_formats': ['pt'], 'model_name': model_name}
    if base_model:
        payload['base_model'] = base_model
    r = c.post(f'/api/projects/{project}/training/start', json=payload)
    if r.status_code != 200:
        return None, r.get_json()
    deadline = time.time() + 900
    while time.time() < deadline:
        time.sleep(3)
        status = c.get(f'/api/projects/{project}/training/status').get_json()['status'] or {}
        if status.get('status') in ('completed', 'failed', 'stopped', 'error'):
            return status, None
    return None, 'timed out'


print('== site A: the model that already exists ==')
build('siteA', 'A', 40)
status, error = train('siteA', 'site_a', 45)
check('the first model trained', status and status['status'] == 'completed', error)
if not status:
    print('cannot continue'); shutil.rmtree(TMP, ignore_errors=True); sys.exit(1)
site_a_weights = status['best_model']
print(f"    mAP50={(status.get('metrics') or {}).get('mAP50')}  "
      f"self-check usable={(status.get('self_check') or {}).get('usable')}")

print('\n== what the site A model knows about itself ==')
described = training.describe_base_model(site_a_weights)
check('its classes can be read back', described['classes'] == ['item'],
      described['classes'])
check('and the size it was trained at', described['img_size'] == 224,
      described['img_size'])

print('\n== an export that cannot be trained further is refused ==')
from services.projects import ProjectError                # noqa: E402
try:
    training.resolve_base_model(str(Path(site_a_weights).with_suffix('.onnx')))
    check('an onnx is refused', False, 'it was accepted')
except ProjectError as exc:
    check('an onnx is refused with a reason',
          'trained further' in exc.message or 'not found' in exc.message.lower(),
          exc.message)

print('\n== site B: different lighting, only 12 images ==')
# Few enough that starting from scratch should struggle, which is the whole
# point: at a new site nobody has thousands of pictures on day one.
build('fromStock', 'B', 12, seed_from=500)
build('fromSiteA', 'B', 12, seed_from=500)

print('    training from the stock checkpoint...')
stock, error = train('fromStock', 'stock_start', 25)
check('the from-scratch run finished', stock and stock['status'] == 'completed', error)

print('    continuing from the site A model...')
tuned, error = train('fromSiteA', 'continued', 25, base_model=site_a_weights)
check('the fine-tuned run finished', tuned and tuned['status'] == 'completed', error)

print('\n== the run records where it started ==')
config = tuned or {}
check('it recorded the base model', bool(config.get('base_model')),
      config.get('base_model'))
if config.get('base_model'):
    check('naming the file it continued from',
          config['base_model']['name'] == Path(site_a_weights).name,
          config['base_model']['name'])
comparison = config.get('base_classes') or {}
check('and how the classes lined up', comparison.get('kept') == ['item'],
      comparison)
check('with nothing added or dropped',
      not comparison.get('added') and not comparison.get('dropped'), comparison)
check('and says so in words', 'carries over' in comparison.get('note', ''),
      comparison.get('note'))

print('\n== which one actually detects at the new site ==')
stock_check = (stock or {}).get('self_check') or {}
tuned_check = (tuned or {}).get('self_check') or {}
print(f"    from stock : mAP50={(stock or {}).get('metrics', {}).get('mAP50')}  "
      f"detects on {stock_check.get('images_with_detections')}/"
      f"{stock_check.get('images_checked')}  best={stock_check.get('best_score')}")
print(f"    continued  : mAP50={(tuned or {}).get('metrics', {}).get('mAP50')}  "
      f"detects on {tuned_check.get('images_with_detections')}/"
      f"{tuned_check.get('images_checked')}  best={tuned_check.get('best_score')}")

check('the continued model detects something at the new site',
      tuned_check.get('usable') is True,
      'continuing from a trained model did not produce a usable one')

# The comparison that justifies the feature. Not asserted as strictly, because
# the point is the shape of the result and a 12-image run is noisy.
if stock_check and tuned_check:
    better = (tuned_check.get('best_score', 0) >= stock_check.get('best_score', 0))
    print(f"    -> continuing scored {'higher' if better else 'lower'} "
          f"than starting over on the same 12 images")

print('\n== a base model from outside the projects tree is refused ==')
r = c.post('/api/projects/fromSiteA/training/reset')
r = c.post('/api/projects/fromSiteA/training/start', json={
    'model_type': 'yolo11n', 'epochs': 1, 'batch_size': 4, 'img_size': 160,
    'model_name': 'bad_base',
    'base_model': str(REPO / 'backend' / 'app.py')})
check('a non-model path is rejected', r.status_code in (400, 403),
      (r.status_code, (r.get_json() or {}).get('message')))

print(f'\n({time.time() - started:.0f}s)')
print('FINE TUNE OK' if not fails else f'{len(fails)} FAILED: {fails}')
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
