"""End-to-end API smoke test against a temporary PROJECTS_ROOT."""
import io, os, sys, tempfile, json, shutil
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix='vt_e2e_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 'test.db').as_posix()
os.environ['ADMIN_PASSWORD'] = 'testpass123'
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import numpy as np, cv2
from app import create_app

app = create_app()
app.config['TESTING'] = True
c = app.test_client()

def j(r):
    try: return r.get_json()
    except Exception: return {'<non-json>': r.data[:200]}

fails = []
def check(label, cond, detail=''):
    print(('  PASS ' if cond else '  FAIL ') + label + ('' if cond else f'  -> {detail}'))
    if not cond: fails.append(label)

print('== health / auth ==')
r = c.get('/api/health'); check('health 200', r.status_code == 200, j(r))

r = c.get('/api/projects')
check('unauthenticated is 401', r.status_code == 401, (r.status_code, j(r)))

r = c.post('/api/auth/login', json={'username': 'admin', 'password': 'wrong'})
check('bad password 401', r.status_code == 401, j(r))

r = c.post('/api/auth/login', json={'username': 'admin', 'password': 'testpass123'})
check('login ok', r.status_code == 200 and j(r)['success'], j(r))

r = c.post('/api/auth/register', json={'username': 'bob', 'email': 'b@x.com', 'password': 'short'})
check('short password rejected', r.status_code == 400, j(r))

print('== project crud + path safety ==')
r = c.post('/api/projects', json={'name': '../evil'})
check('traversal name rejected', r.status_code == 400, j(r))

r = c.post('/api/projects', json={'name': 'e2e', 'description': 'test'})
check('create project', r.status_code == 200 and j(r)['success'], j(r))

r = c.post('/api/projects', json={'name': 'e2e'})
check('duplicate rejected', r.status_code == 400, j(r))

r = c.get('/api/projects')
check('list has 1', len(j(r)['projects']) == 1, j(r))

print('== image upload ==')
def make_png(w, h, seed):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 40, np.uint8)
    boxes = []
    for i in range(2):
        x = int(rng.integers(10, w - 60)); y = int(rng.integers(10, h - 60))
        bw = int(rng.integers(30, 55)); bh = int(rng.integers(30, 55))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 220, 0) if i == 0 else (220, 0, 0), -1)
        boxes.append((f'cls{i}', x, y, bw, bh))
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes(), boxes

uploads, expected = [], []
for i in range(14):
    data, boxes = make_png(200, 160, i)
    uploads.append((io.BytesIO(data), f'img{i}.png'))
    expected.append(boxes)

r = c.post('/api/projects/e2e/images', data={'images': uploads},
           content_type='multipart/form-data')
res = j(r)
check('imported 14', res.get('imported_count') == 14, res)
names = [x['filename'] for x in res['imported']]

r = c.get('/api/projects/e2e/images')
imgs = j(r)['images']
check('dimensions present', all(i['width'] == 200 and i['height'] == 160 for i in imgs),
      imgs[:1])

print('== annotations ==')
for name, boxes in zip(names, expected):
    regions = [{'tag': t, 'x': x, 'y': y, 'width': w, 'height': h} for t, x, y, w, h in boxes]
    r = c.post(f'/api/projects/e2e/images/{name}/annotations', json={'regions': regions})
    assert j(r)['success'], j(r)

# out-of-bounds box must be clamped, degenerate box dropped
r = c.post(f'/api/projects/e2e/images/{names[0]}/annotations', json={'regions': [
    {'tag': 'cls0', 'x': 10, 'y': 10, 'width': 40, 'height': 40},
    {'tag': 'cls1', 'x': 190, 'y': 150, 'width': 500, 'height': 500},
    {'tag': 'cls0', 'x': 5, 'y': 5, 'width': 0.2, 'height': 0.2},
    {'tag': '', 'x': 1, 'y': 1, 'width': 10, 'height': 10},
]})
check('degenerate/untagged dropped', j(r)['saved_count'] == 2, j(r))
ann = json.loads((Path(os.environ['PROJECTS_ROOT']) / 'e2e' / 'annotations' /
                  f'{names[0]}.json').read_text(encoding='utf-8'))
clamped = ann['regions'][1]
check('box clamped to image',
      clamped['x'] + clamped['width'] <= 200 and clamped['y'] + clamped['height'] <= 160,
      clamped)

print('== gallery carries ROI boxes ==')
imgs = c.get('/api/projects/e2e/images').get_json()['images']
annotated = [i for i in imgs if i['annotated']]
check('every annotated image ships its boxes',
      all(i.get('boxes') for i in annotated), len(annotated))
check('unannotated images ship none',
      all(not i.get('boxes') for i in imgs if not i['annotated']), True)
# The overlay draws in image coordinates, so a box outside the frame would
# render outside the thumbnail.
outside = [(i['filename'], b) for i in annotated for b in i['boxes']
           if b[0] < 0 or b[1] < 0
           or b[0] + b[2] > i['width'] or b[1] + b[3] > i['height']]
check('no box falls outside its image', not outside, outside[:2])
check('each box carries its class name',
      all(isinstance(b[4], str) and b[4] for i in annotated for b in i['boxes']))

r = c.get('/api/projects/e2e/dataset-summary')
s = j(r)
check('summary classes', s['num_classes'] == 2 and s['annotated_images'] == 14, s)
check('readiness computed', 0 <= s['readiness_score'] <= 100, s['readiness_score'])

print('== dataset build (the critical path) ==')
r = c.post('/api/projects/e2e/prepare-dataset')
d = j(r)['dataset']
check('train+val covers all', d['train_images'] + d['val_images'] >= 14, d)
check('boxes written', d['train_boxes'] > 0 and d['val_boxes'] > 0, d)

label_dir = Path(d['dataset_path']) / 'labels' / 'train'
label_files = list(label_dir.glob('*.txt'))
check('label files exist', len(label_files) == d['train_images'], len(label_files))

bad = []
for lf in label_files:
    for line in lf.read_text().strip().splitlines():
        parts = line.split()
        cid = int(parts[0]); vals = [float(v) for v in parts[1:]]
        if cid not in (0, 1) or any(not (0.0 <= v <= 1.0) for v in vals) or \
           vals[2] <= 0 or vals[3] <= 0:
            bad.append((lf.name, line))
check('all YOLO coords normalized in [0,1]', not bad, bad[:3])

import yaml
cfg = yaml.safe_load(Path(d['data_yaml']).read_text(encoding='utf-8'))
check('data.yaml names', cfg['nc'] == 2 and cfg['names'] == {0: 'cls0', 1: 'cls1'}, cfg)
check('data.yaml abs path', Path(cfg['path']).is_absolute(), cfg['path'])

# verify a label round-trips back to the right pixel box
sample = label_files[0]
img_name = None
for n in names:
    if Path(n).stem == sample.stem: img_name = n
line = sample.read_text().strip().splitlines()[0]
_, cx, cy, nw, nh = line.split()
px_w, px_h = float(nw) * 200, float(nh) * 160
src = json.loads((Path(os.environ['PROJECTS_ROOT']) / 'e2e' / 'annotations' /
                  f'{img_name}.json').read_text(encoding='utf-8'))
src_dims = sorted([(r['width'], r['height']) for r in src['regions']])
check('label size round-trips to pixels',
      any(abs(px_w - w) < 1.5 and abs(px_h - h) < 1.5 for w, h in src_dims),
      (px_w, px_h, src_dims))

print('== augmentation ==')
r = c.post('/api/projects/e2e/augment-color',
           json={'tones': ['gray', 'clahe'], 'variants_per_tone': 1})
a = j(r)
check('augment created', a['created_count'] == 28, a.get('created_count'))
print('== gallery carries ROI boxes ==')
imgs = c.get('/api/projects/e2e/images').get_json()['images']
annotated = [i for i in imgs if i['annotated']]
check('every annotated image ships its boxes',
      all(i.get('boxes') for i in annotated), len(annotated))
check('unannotated images ship none',
      all(not i.get('boxes') for i in imgs if not i['annotated']), True)
# The overlay draws in image coordinates, so a box outside the frame would
# render outside the thumbnail.
outside = [(i['filename'], b) for i in annotated for b in i['boxes']
           if b[0] < 0 or b[1] < 0
           or b[0] + b[2] > i['width'] or b[1] + b[3] > i['height']]
check('no box falls outside its image', not outside, outside[:2])
check('each box carries its class name',
      all(isinstance(b[4], str) and b[4] for i in annotated for b in i['boxes']))

r = c.get('/api/projects/e2e/dataset-summary')
check('images grew to 42', j(r)['total_images'] == 42, j(r)['total_images'])

r = c.post('/api/projects/e2e/prepare-dataset')
d2 = j(r)['dataset']
val_names = {p.name for p in (Path(d2['dataset_path']) / 'images' / 'val').iterdir()}
check('no augmented images in val split',
      not any('_aug_' in n for n in val_names), sorted(val_names)[:3])
check('rebuild is clean (no stale labels)',
      len(list((Path(d2['dataset_path']) / 'labels' / 'train').glob('*.txt'))) == d2['train_images'])

print('== export / import zip ==')
r = c.post('/api/projects/e2e/export')
check('export zip', r.status_code == 200 and r.data[:2] == b'PK', r.status_code)
zip_bytes = r.data
c.post('/api/projects', json={'name': 'e2e2'})
r = c.post('/api/projects/e2e2/import-dataset',
           data={'file': (io.BytesIO(zip_bytes), 'd.zip')},
           content_type='multipart/form-data')
check('import zip', j(r).get('imported_images') == 42, j(r))

print('== training validation ==')
r = c.post('/api/projects/e2e/training/start', json={'model_type': 'nope'})
check('bad model type rejected', r.status_code == 400, j(r))
r = c.post('/api/projects/e2e/training/start', json={'model_type': 'yolo11n', 'img_size': 500})
check('non-multiple-of-32 imgsz rejected', r.status_code == 400, j(r))
r = c.post('/api/projects/e2e/training/start', json={'model_type': 'yolo11n', 'epochs': 0})
check('epochs 0 rejected', r.status_code == 400, j(r))
r = c.post('/api/projects/e2e/training/start',
           json={'model_type': 'yolo11n', 'export_formats': ['exe']})
check('bad export format rejected', r.status_code == 400, j(r))

r = c.get('/api/projects/e2e/training/status')
check('status before any run', j(r)['has_run'] is False, j(r))

print('== model download path safety ==')
r = c.get('/api/projects/e2e/training/models/download?path=' +
          str(Path(os.environ['PROJECTS_ROOT']).parent / 'test.db'))
check('outside-project download blocked', r.status_code == 403, r.status_code)

r = c.get('/api/overview')
o = j(r)
check('overview aggregates', o['project_count'] == 2 and o['total_images'] == 84, o)

print('\n' + ('ALL PASSED' if not fails else f'{len(fails)} FAILED: {fails}'))
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
