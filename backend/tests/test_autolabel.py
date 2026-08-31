"""Train a small model, then use it to auto-label unannotated images."""
import io, os, sys, time, tempfile, shutil, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix='vt_auto_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import numpy as np, cv2
from app import create_app

app = create_app()
c = app.test_client()
c.post('/api/projects', json={'name': 'auto'})

fails = []
def check(label, cond, detail=''):
    print(('  PASS ' if cond else '  FAIL ') + label + ('' if cond else f'  -> {detail}'))
    if not cond: fails.append(label)

def make(seed):
    rng = np.random.default_rng(seed)
    img = np.full((256, 256, 3), 28, np.uint8)
    # Some texture so the detector has more than one cue to learn.
    noise = rng.integers(0, 26, (256, 256, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    x, y = int(rng.integers(30, 160)), int(rng.integers(30, 160))
    cv2.rectangle(img, (x, y), (x + 64, y + 64), (0, 235, 90), -1)
    cv2.rectangle(img, (x + 12, y + 12), (x + 52, y + 52), (20, 40, 20), 3)
    cv2.circle(img, (x + 32, y + 32), 10, (255, 255, 255), -1)
    return cv2.imencode('.png', img)[1].tobytes(), (x, y)

# 30 images: 20 hand-labelled for training, 10 left blank for the model.
uploads, meta = [], []
for i in range(30):
    blob, xy = make(i)
    uploads.append((io.BytesIO(blob), f'i{i:02}.png'))
    meta.append(xy)

r = c.post('/api/projects/auto/images', data={'images': uploads},
           content_type='multipart/form-data')
names = [x['filename'] for x in r.get_json()['imported']]
LABELLED = 20
for name, (x, y) in list(zip(names, meta))[:LABELLED]:
    c.post(f'/api/projects/auto/images/{name}/annotations',
           json={'regions': [{'tag': 'block', 'x': x, 'y': y, 'width': 64, 'height': 64}]})

summary = c.get('/api/projects/auto/dataset-summary').get_json()
check('20 of 30 annotated to start', summary['annotated_images'] == 20, summary['annotated_images'])

print('\n== auto-label before any training is refused with a clear reason ==')
r = c.post('/api/projects/auto/auto-label', json={})
message = (r.get_json() or {}).get('message', '')
check('refused', r.status_code == 400, r.status_code)
# The wording differs depending on whether any other project has a model to
# borrow, so the check is that it says what to do rather than that it says a
# particular sentence.
check('and says what to do about it',
      'train' in message.lower() and len(message) > 40, message)

print('\n== a model from outside the projects tree is still refused ==')
r = c.post('/api/projects/auto/auto-label',
           json={'model_path': str(BACKEND / 'app.py')})
check('a non-model path is rejected', r.status_code in (400, 403),
      (r.status_code, (r.get_json() or {}).get('message')))

print('\n== train a small model ==')
r = c.post('/api/projects/auto/training/start', json={
    'model_type': 'yolo11n', 'epochs': 40, 'batch_size': 4, 'img_size': 320,
    'export_formats': ['pt'], 'model_name': 'seed'})
if not r.get_json().get('success'):
    print('  START FAILED:', r.get_json()); shutil.rmtree(TMP, ignore_errors=True); sys.exit(1)

deadline = time.time() + 900
while time.time() < deadline:
    time.sleep(6)
    st = c.get('/api/projects/auto/training/status').get_json()['status']
    if st['status'] in ('completed', 'failed', 'stopped'):
        break
print(f"  training {st['status']}, mAP50={st.get('metrics', {}).get('mAP50')}")
check('training completed', st['status'] == 'completed', st.get('error'))

print('\n== auto-label the remaining 10 ==')
r = c.post('/api/projects/auto/auto-label', json={'score_threshold': 0.25})
job = r.get_json().get('job')
check('job accepted', r.status_code == 200, r.get_json())
check('targets only the unannotated', job and job['total'] == 10, job.get('total') if job else None)
print(f"  using model: {Path(job['model']).name}")

deadline = time.time() + 300
while time.time() < deadline:
    time.sleep(2)
    j = c.get('/api/projects/auto/auto-label').get_json()['job']
    if j['status'] != 'running':
        break
print(f"  {j['status']}: processed={j['processed']} labelled={j['labelled']} "
      f"boxes={j['boxes']} skipped={j['skipped']}")
for e in (j.get('errors') or []): print('   ERROR:', e)
check('job completed', j['status'] == 'completed', j.get('error'))
check('it labelled something', j['labelled'] > 0, j)

print('\n== trying it without writing anything ==')
# The pass only touches unlabelled images, so on a project that is already
# fully annotated there was no way to find out whether a model was worth using
# short of letting it overwrite work drawn by hand.
def region_counts(subset):
    return {
        name: len(json.loads(
            (Path(os.environ['PROJECTS_ROOT']) / 'auto' / 'annotations' /
             f'{name}.json').read_text(encoding='utf-8')).get('regions', []))
        for name in subset
    }


before_counts = region_counts(names[:LABELLED])

r = c.post('/api/projects/auto/auto-label/preview',
           json={'score_threshold': 0.25, 'sample': 4})
preview = r.get_json()
check('the preview ran', r.status_code == 200, preview)
check('it looked at some images', len(preview.get('results', [])) > 0, preview)
check('it names the model it used', bool(preview.get('model_name')),
      preview.get('model_name'))
check('and gives a verdict in words', len(preview.get('verdict', '')) > 30,
      preview.get('verdict'))

rows = preview.get('results', [])
check('it prefers images drawn by hand, so there is something to compare against',
      all(row.get('drawn_by_hand') for row in rows),
      [(row['filename'], row.get('drawn_by_hand')) for row in rows])
check('and reports what the model found on each',
      all('model_found' in row for row in rows), rows[:1])

check('nothing on disk changed', region_counts(names[:LABELLED]) == before_counts,
      [k for k, v in region_counts(names[:LABELLED]).items()
       if before_counts[k] != v])

print('\n== the predictions are real annotations ==')
after = c.get('/api/projects/auto/dataset-summary').get_json()
check('annotated count grew', after['annotated_images'] > 20, after['annotated_images'])

images = c.get('/api/projects/auto/images').get_json()['images']
auto = [i for i in images if i['annotated'] and i['filename'] in names[LABELLED:]]
check('previously blank images now have boxes', len(auto) == j['labelled'], len(auto))

# spot-check geometry against ground truth
ok_boxes = 0
for idx, name in enumerate(names[LABELLED:], start=LABELLED):
    ann = json.loads((Path(os.environ['PROJECTS_ROOT']) / 'auto' / 'annotations' /
                      f'{name}.json').read_text(encoding='utf-8'))
    if not ann.get('regions'):
        continue
    gx, gy = meta[idx]
    r0 = ann['regions'][0]
    if abs(r0['x'] - gx) < 25 and abs(r0['y'] - gy) < 25:
        ok_boxes += 1
    check_marked = ann.get('auto_labelled') is True
print(f'  {ok_boxes}/{j["labelled"]} predicted boxes land within 25px of ground truth')
check('predictions are geometrically sensible', ok_boxes >= max(1, j['labelled'] // 2),
      f'{ok_boxes}/{j["labelled"]}')

sample = json.loads((Path(os.environ['PROJECTS_ROOT']) / 'auto' / 'annotations' /
                     f'{names[LABELLED]}.json').read_text(encoding='utf-8'))
check('marked as auto-labelled for review', sample.get('auto_labelled') is True, sample.keys())
check('records which model produced it', bool(sample.get('auto_label', {}).get('model')))

print('\n== the result is trainable ==')
r = c.post('/api/projects/auto/prepare-dataset')
check('dataset builds from mixed hand + auto labels', r.status_code == 200, r.get_json())
if r.status_code == 200:
    d = r.get_json()['dataset']
    print(f"  {d['train_images']} train / {d['val_images']} val, "
          f"{d['train_boxes']} / {d['val_boxes']} boxes")

print('\n' + ('AUTO-LABEL OK' if not fails else f'{len(fails)} FAILED: {fails}'))
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
