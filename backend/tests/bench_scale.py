"""Timing against a project the size of the user's real one (2232 images)."""
import io, os, sys, tempfile, shutil, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix='vt_perf_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import numpy as np, cv2
from app import create_app

N = 2232
CLASSES = [str(d) for d in range(10)]

app = create_app()
c = app.test_client()
c.post('/api/projects', json={'name': 'big'})

print(f'building a {N}-image project on disk...')
root = Path(os.environ['PROJECTS_ROOT']) / 'big'
(root / 'images').mkdir(parents=True, exist_ok=True)
(root / 'annotations').mkdir(parents=True, exist_ok=True)

# Written directly rather than through the upload endpoint: this is about
# measuring the read paths, not the uploader.
img = np.full((480, 640, 3), 40, np.uint8)
cv2.rectangle(img, (100, 100), (200, 220), (0, 200, 0), -1)
ok, buf = cv2.imencode('.jpg', img)
blob = buf.tobytes()

t0 = time.time()
for i in range(N):
    name = f'img_{i:05d}.jpg'
    (root / 'images' / name).write_bytes(blob)
    regions = [{'tag': CLASSES[i % 10], 'x': 100, 'y': 100, 'width': 100, 'height': 120}]
    if i % 3 == 0:
        regions.append({'tag': CLASSES[(i + 3) % 10], 'x': 300, 'y': 150, 'width': 90, 'height': 90})
    (root / 'annotations' / f'{name}.json').write_text(json.dumps({
        'filename': name, 'regions': regions, 'annotated': True,
        'width': 640, 'height': 480,
    }, ensure_ascii=False), encoding='utf-8')
print(f'  built in {time.time() - t0:.1f}s')

def timed(label, fn, budget):
    start = time.time()
    result = fn()
    elapsed = time.time() - start
    verdict = 'ok  ' if elapsed <= budget else 'SLOW'
    print(f'  [{verdict}] {label:38} {elapsed * 1000:8.0f} ms   (budget {budget * 1000:.0f} ms)')
    return elapsed, result

print(f'\n== read paths on {N} images ==')
timed('GET /projects  (list)', lambda: c.get('/api/projects'), 0.5)
timed('GET /projects/big', lambda: c.get('/api/projects/big'), 0.2)
t_cold, r = timed('GET /images  (cold: builds index)', lambda: c.get('/api/projects/big/images'), 30.0)
t_images, r = timed('GET /images  (warm)', lambda: c.get('/api/projects/big/images'), 1.0)
timed('GET /images  (warm, again)', lambda: c.get('/api/projects/big/images'), 1.0)
print(f'         -> {len(r.get_json()["images"])} images, '
      f'{len(r.data) / 1024:.0f} KB of JSON')
t_sum, _ = timed('GET /dataset-summary', lambda: c.get('/api/projects/big/dataset-summary'), 3.0)

print('\n== the write path an annotator hits constantly ==')
target = 'img_00000.jpg'
body = {'regions': [{'tag': '1', 'x': 10, 'y': 10, 'width': 50, 'height': 50}]}
timed('POST annotations (first, cold)',
      lambda: c.post(f'/api/projects/big/images/{target}/annotations', json=body), 3.0)

samples = []
for i in range(10):
    start = time.time()
    c.post(f'/api/projects/big/images/img_{i:05d}.jpg/annotations', json=body)
    samples.append(time.time() - start)
avg = sum(samples) / len(samples)
worst = max(samples)
verdict = 'ok  ' if avg <= 0.5 else 'SLOW'
print(f'  [{verdict}] POST annotations (avg of 10)           {avg * 1000:8.0f} ms   '
      f'(worst {worst * 1000:.0f} ms)')

print('\n== dataset build ==')
t_build, r = timed('POST /prepare-dataset', lambda: c.post('/api/projects/big/prepare-dataset'), 60.0)
d = r.get_json()
if d.get('success'):
    rep = d['dataset']
    print(f'         -> {rep["train_images"]} train / {rep["val_images"]} val, '
          f'{rep["train_boxes"]} / {rep["val_boxes"]} boxes, '
          f'{len(rep["classes"])} classes')

print('\n== verdict ==')
problems = []
if avg > 0.5:
    problems.append(f'saving one annotation takes {avg * 1000:.0f} ms — an annotator '
                    f'saves constantly, so this is the number that decides whether the '
                    f'tool feels usable')
if t_images > 1.0:
    problems.append(f'the warm gallery request takes {t_images:.1f}s')
if t_sum > 3.0:
    problems.append(f'dataset-summary takes {t_sum:.1f}s')

for p in problems:
    print('  PROBLEM:', p)
if not problems:
    print('  all paths within budget')

shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if problems else 0)
