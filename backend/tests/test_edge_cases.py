"""
Edge cases the tidy synthetic tests do not reach:
mixed image formats and sizes, non-ASCII names, huge/tiny images, corrupt files,
a class with one example, boxes at the border, unicode class names, concurrency.
"""
import io, os, sys, tempfile, shutil, json, threading
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix='vt_edge_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import numpy as np, cv2
from app import create_app

app = create_app()
c = app.test_client()

fails = []
def check(label, cond, detail=''):
    print(('  PASS ' if cond else '  FAIL ') + label + ('' if cond else f'  -> {detail}'))
    if not cond: fails.append(label)

def encode(w, h, ext='.png', colour=(0, 200, 0)):
    img = np.full((h, w, 3), 40, np.uint8)
    cv2.rectangle(img, (w // 4, h // 4), (w // 2, h // 2), colour, -1)
    ok, buf = cv2.imencode(ext, img)
    return buf.tobytes()

print('== project names ==')
for bad in ['../escape', 'a/b', '..', '', 'x' * 100, 'con/../..']:
    r = c.post('/api/projects', json={'name': bad})
    if r.status_code != 400:
        check(f'rejects {bad!r}', False, r.status_code)
check('rejects traversal / oversized names', True)

r = c.post('/api/projects', json={'name': 'โปรเจกต์ ทดสอบ'})
check('accepts a Thai project name', r.status_code == 200, r.get_json())
THAI = 'โปรเจกต์ ทดสอบ'

r = c.post('/api/projects', json={'name': 'edge'})
check('creates plain project', r.status_code == 200, r.get_json())

print('\n== mixed formats, sizes, and bad files ==')
uploads = [
    (io.BytesIO(encode(640, 480, '.jpg')), 'a.jpg'),
    (io.BytesIO(encode(800, 600, '.png')), 'b.png'),
    (io.BytesIO(encode(1920, 1080, '.bmp')), 'c.bmp'),
    (io.BytesIO(encode(64, 48, '.webp')), 'd.webp'),
    (io.BytesIO(encode(300, 300, '.jpg')), 'ภาพไทย.jpg'),
    (io.BytesIO(b'this is not an image at all'), 'broken.jpg'),
    (io.BytesIO(b''), 'empty.png'),
    (io.BytesIO(encode(100, 100, '.jpg')), 'script.exe'),
]
r = c.post('/api/projects/edge/images', data={'images': uploads},
           content_type='multipart/form-data')
res = r.get_json()
print('   imported:', res['imported_count'], '| rejected:', [x['reason'] for x in res['rejected']])
check('imports the 5 valid images', res['imported_count'] == 5, res['imported_count'])
check('rejects corrupt/empty/wrong-extension', len(res['rejected']) == 3, res['rejected'])

imgs = c.get('/api/projects/edge/images').get_json()['images']
dims = sorted((i['width'], i['height']) for i in imgs)
check('records true dimensions for every format',
      dims == [(64, 48), (300, 300), (640, 480), (800, 600), (1920, 1080)], dims)

print('\n== annotation edge cases ==')
by_size = {(i['width'], i['height']): i['filename'] for i in imgs}
big = by_size[(1920, 1080)]
tiny = by_size[(64, 48)]

# box exactly on the border, box larger than the image, sub-pixel box
r = c.post(f'/api/projects/edge/images/{big}/annotations', json={'regions': [
    {'tag': 'ป้าย', 'x': 0, 'y': 0, 'width': 1920, 'height': 1080},          # full frame
    {'tag': 'ป้าย', 'x': 1900, 'y': 1060, 'width': 500, 'height': 500},      # overflows
    {'tag': 'edge', 'x': 1919.6, 'y': 1079.6, 'width': 0.3, 'height': 0.3},  # degenerate
]})
saved = r.get_json()
check('keeps the 2 usable boxes, drops the degenerate one', saved['saved_count'] == 2, saved)

ann = json.loads((Path(os.environ['PROJECTS_ROOT']) / 'edge' / 'annotations' / f'{big}.json')
                 .read_text(encoding='utf-8'))
inside = all(r['x'] >= 0 and r['y'] >= 0 and r['x'] + r['width'] <= 1920
             and r['y'] + r['height'] <= 1080 for r in ann['regions'])
check('every stored box lies inside the image', inside, ann['regions'])

c.post(f'/api/projects/edge/images/{tiny}/annotations', json={'regions': [
    {'tag': 'ป้าย', 'x': 5, 'y': 5, 'width': 20, 'height': 20}]})
for filename in [i['filename'] for i in imgs if i['filename'] not in (big, tiny)]:
    c.post(f'/api/projects/edge/images/{filename}/annotations', json={'regions': [
        {'tag': 'ป้าย', 'x': 10, 'y': 10, 'width': 30, 'height': 30}]})

s = c.get('/api/projects/edge/dataset-summary').get_json()
check('unicode class name survives', 'ป้าย' in s['tags'], list(s['tags']))
check('the single-example class is counted', s['num_classes'] == 1, s)

print('\n== dataset build with mixed sizes ==')
r = c.post('/api/projects/edge/prepare-dataset')
d = r.get_json()
check('build succeeds', r.status_code == 200 and d.get('success'), d)
if d.get('success'):
    rep = d['dataset']
    label_root = Path(rep['dataset_path']) / 'labels'
    bad = []
    for split in ('train', 'val'):
        for lf in (label_root / split).glob('*.txt'):
            for line in lf.read_text(encoding='utf-8').strip().splitlines():
                parts = line.split()
                vals = [float(v) for v in parts[1:]]
                if any(not (0.0 <= v <= 1.0) for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                    bad.append((split, lf.name, line))
    check('all labels normalised despite differing image sizes', not bad, bad[:3])

    import yaml
    cfg = yaml.safe_load(Path(rep['data_yaml']).read_text(encoding='utf-8'))
    check('data.yaml keeps the unicode class name', cfg['names'] == {0: 'ป้าย'}, cfg['names'])

print('\n== unicode project end-to-end ==')
r = c.post(f'/api/projects/{THAI}/images',
           data={'images': [(io.BytesIO(encode(200, 200, '.jpg')), 'x.jpg')]},
           content_type='multipart/form-data')
check('uploads into a Thai-named project', r.get_json().get('imported_count') == 1, r.get_json())
fn = r.get_json()['imported'][0]['filename']
r = c.get(f'/api/projects/{THAI}/images/{fn}')
check('reads image data back (cv2 unicode path)', r.get_json().get('success'), r.get_json())
r = c.get(f'/api/projects/{THAI}/images/{fn}/raw')
check('serves raw bytes', r.status_code == 200 and len(r.data) > 100, r.status_code)

print('\n== deleting an image keeps stats honest ==')
before = c.get('/api/projects/edge/dataset-summary').get_json()
c.delete(f'/api/projects/edge/images/{tiny}')
after = c.get('/api/projects/edge/dataset-summary').get_json()
check('image count drops by one', after['total_images'] == before['total_images'] - 1,
      (before['total_images'], after['total_images']))
check('box count drops too', after['total_boxes'] < before['total_boxes'],
      (before['total_boxes'], after['total_boxes']))

print('\n== concurrent annotation writes ==')
target = [i['filename'] for i in c.get('/api/projects/edge/images').get_json()['images']][0]
errors = []
def hammer(n):
    try:
        cl = app.test_client()
        for i in range(12):
            cl.post(f'/api/projects/edge/images/{target}/annotations', json={'regions': [
                {'tag': 'ป้าย', 'x': n, 'y': i, 'width': 20, 'height': 20}]})
    except Exception as e:
        errors.append(repr(e))

threads = [threading.Thread(target=hammer, args=(n,)) for n in range(6)]
[t.start() for t in threads]; [t.join() for t in threads]
check('no exceptions under concurrent writes', not errors, errors[:2])
final = c.get(f'/api/projects/edge/images/{target}').get_json()
check('annotation file is still valid JSON afterwards', final.get('success'), final)

print('\n== zip import safety ==')
import zipfile
evil = TMP / 'evil.zip'
with zipfile.ZipFile(evil, 'w') as z:
    z.writestr('dataset.json', json.dumps({'annotations': [
        {'filename': '../../../../escaped.jpg', 'regions': [], 'annotated': False}]}))
    z.writestr('images/../../../../escaped.jpg', encode(50, 50, '.jpg'))
r = c.post('/api/projects/edge/import-dataset',
           data={'file': (open(evil, 'rb'), 'evil.zip')},
           content_type='multipart/form-data')
escaped = (TMP.parent / 'escaped.jpg').exists() or (TMP / 'escaped.jpg').exists()
check('zip traversal writes nothing outside the project', not escaped)

print('\n== oversized / malformed requests ==')
r = c.post('/api/projects/edge/images/nope.jpg/annotations', json={'regions': []})
check('annotating a missing image is a clean 404', r.status_code == 404, r.status_code)
r = c.post('/api/projects/edge/images/../../x/annotations', json={'regions': []})
check('traversal in filename is rejected', r.status_code in (400, 404), r.status_code)
r = c.get('/api/projects/does-not-exist/images')
check('missing project is 404', r.status_code == 404, r.status_code)
r = c.post('/api/projects/edge/augment-color', json={'variants_per_tone': 99999})
check('absurd augment count is clamped or refused', r.status_code in (200, 400), r.status_code)

print('\n' + ('EDGE CASES OK' if not fails else f'{len(fails)} FAILED: {fails}'))
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
