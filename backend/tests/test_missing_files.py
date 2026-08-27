"""
The recovery path: a project whose image files vanished, then come back.

This is exactly the state test01 is in, so it is worth proving end to end
rather than reasoning about.
"""
import io, os, sys, json, shutil, tempfile
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix='vt_recover_'))
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

c.post('/api/projects', json={'name': 'recover'})
root = Path(os.environ['PROJECTS_ROOT']) / 'recover'

uploads, truth = [], []
for i in range(8):
    img = np.full((300, 400, 3), 30, np.uint8)
    x, y = 40 + i * 12, 60
    cv2.rectangle(img, (x, y), (x + 70, y + 90), (0, 220, 90), -1)
    uploads.append((io.BytesIO(cv2.imencode('.jpg', img)[1].tobytes()), f'r{i}.jpg'))
    truth.append((x, y))

r = c.post('/api/projects/recover/images', data={'images': uploads},
           content_type='multipart/form-data')
names = [x['filename'] for x in r.get_json()['imported']]
for name, (x, y) in zip(names, truth):
    c.post(f'/api/projects/recover/images/{name}/annotations',
           json={'regions': [{'tag': 'blob', 'x': x, 'y': y, 'width': 70, 'height': 90}]})

before = c.get('/api/projects/recover/images').get_json()['images']
check('8 images listed with boxes', len(before) == 8 and all(i['boxes'] for i in before),
      len(before))

print('\n== the image folder disappears (drive unplugged, folder moved) ==')
stash = TMP / 'stash'
shutil.move(str(root / 'images'), str(stash))

proj = c.get('/api/projects/recover').get_json()['project']
check('counters are preserved, not zeroed', proj['total_images'] == 8, proj['total_images'])
check('the API reports the files as unavailable', proj['images_available'] is False)
listed = c.get('/api/projects/recover/images').get_json()['images']
check('gallery lists nothing', len(listed) == 0, len(listed))

print('\n== "Check again" while they are still gone ==')
r = c.post('/api/projects/recover/rescan', json={})
body = r.get_json()
check('reports still missing', body['images_available'] is False, body)
check('and leaves the counts alone',
      c.get('/api/projects/recover').get_json()['project']['total_images'] == 8)
print(f"    message: {body['message'][:88]}")

print('\n== the files come back, then "Check again" ==')
shutil.move(str(stash), str(root / 'images'))
r = c.post('/api/projects/recover/rescan', json={})
body = r.get_json()
check('reports them found', body['images_available'] is True, body)
check('all 8 counted again', body['total_images'] == 8, body.get('total_images'))
print(f"    message: {body['message']}")

after = c.get('/api/projects/recover/images').get_json()['images']
check('gallery lists them again', len(after) == 8, len(after))
check('ROI boxes are back', all(i['boxes'] for i in after))
check('boxes match what was drawn originally',
      sorted(tuple(b[:4]) for i in after for b in i['boxes'])
      == sorted(tuple(b[:4]) for i in before for b in i['boxes']))

print('\n== raw image bytes are served again ==')
r = c.get(f'/api/projects/recover/images/{after[0]["filename"]}/raw')
served_ok = r.status_code == 200 and len(r.data) > 500
# Windows keeps the file locked until the response is closed, so this has to
# happen before the folder can be moved again.
r.close()
check('raw endpoint returns an image', served_ok, f'{r.status_code} {len(r.data)} bytes')

print('\n== "Clear the counts" when they are gone for good ==')
shutil.move(str(root / 'images'), str(stash))
r = c.post('/api/projects/recover/rescan', json={'clear_if_missing': True})
body = r.get_json()
check('counts reset to zero', body['total_images'] == 0, body)
proj = c.get('/api/projects/recover').get_json()['project']
check('project now reports an empty dataset',
      proj['total_images'] == 0 and proj['total_annotations'] == 0, proj)
print(f"    message: {body['message']}")

print('\n== and importing starts it over cleanly ==')
img = np.full((200, 200, 3), 30, np.uint8)
cv2.rectangle(img, (20, 20), (90, 90), (0, 220, 90), -1)
r = c.post('/api/projects/recover/images',
           data={'images': [(io.BytesIO(cv2.imencode('.jpg', img)[1].tobytes()), 'new.jpg')]},
           content_type='multipart/form-data')
check('import works after clearing', r.get_json().get('imported_count') == 1, r.get_json())
check('project reports exactly the new image',
      c.get('/api/projects/recover').get_json()['project']['total_images'] == 1)

print('\n' + ('RECOVERY OK' if not fails else f'{len(fails)} FAILED: {fails}'))
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
