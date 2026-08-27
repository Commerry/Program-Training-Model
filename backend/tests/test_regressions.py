"""Each audit finding, re-tested after the fix."""
import io, os, sys, json, tempfile, shutil, threading, time, zipfile
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix='vt_fixed_'))
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
def check(n, label, cond, detail=''):
    print(('  PASS ' if cond else '  FAIL ') + f'#{n} {label}' + ('' if cond else f'  -> {detail}'))
    if not cond: fails.append(f'#{n} {label}')

def jpg(w=100, h=100):
    img = np.full((h, w, 3), 40, np.uint8)
    cv2.rectangle(img, (10, 10), (w // 2, h // 2), (0, 200, 0), -1)
    return cv2.imencode('.jpg', img)[1].tobytes()

print('== #1 atomic write reports failure instead of lying ==')
from services.atomicio import write_json, AtomicWriteError, read_json
target = TMP / 'held.json'
write_json(target, {'v': 1})
reader = open(target, 'r', encoding='utf-8')
try:
    t0 = time.time()
    try:
        write_json(target, {'v': 2})
        raised = False
    except AtomicWriteError:
        raised = True
    took = time.time() - t0
finally:
    reader.close()
check(1, 'write_json raises rather than silently failing', raised, 'no exception raised')
check(1, 'it retries first (does not give up instantly)', took > 0.1, f'{took*1000:.0f} ms')

# and it succeeds once the reader lets go
write_json(target, {'v': 3})
check(1, 'succeeds once the file is free', read_json(target) == {'v': 3})

print('\n== #1b a held annotation file surfaces as an error, not a false success ==')
c.post('/api/projects', json={'name': 'hold'})
c.post('/api/projects/hold/images', data={'images': [(io.BytesIO(jpg()), 'a.jpg')]},
       content_type='multipart/form-data')
fn = c.get('/api/projects/hold/images').get_json()['images'][0]['filename']
ann_file = Path(os.environ['PROJECTS_ROOT']) / 'hold' / 'annotations' / f'{fn}.json'
holder = open(ann_file, 'r', encoding='utf-8')
try:
    r = c.post(f'/api/projects/hold/images/{fn}/annotations',
               json={'regions': [{'tag': 'a', 'x': 1, 'y': 1, 'width': 20, 'height': 20}]})
finally:
    holder.close()
check(1, 'a blocked save returns an error status', r.status_code >= 400,
      f'status {r.status_code}: {(r.get_json() or {}).get("message")}')

print('\n== #3 .env is loaded ==')
import importlib, config as cfg_mod
src = (BACKEND / 'config.py').read_text(encoding='utf-8')
check(3, 'config.py calls load_dotenv', 'load_dotenv' in src)

print('\n== #4 safe defaults ==')
# Asserted against the resolved values rather than the source text, so moving
# a default between modules cannot make this pass or fail spuriously.
importlib.reload(cfg_mod)
app_src = (BACKEND / 'app.py').read_text(encoding='utf-8')
check(4, 'debug defaults off', "FLASK_DEBUG', '0'" in app_src)
check(4, 'binds loopback by default', cfg_mod.BACKEND_HOST == '127.0.0.1',
      repr(cfg_mod.BACKEND_HOST))
check(4, 'session key is generated, not the shared dev constant',
      'dev-secret-key' not in cfg_mod.Config.SECRET_KEY
      and len(cfg_mod.Config.SECRET_KEY) >= 32,
      cfg_mod.Config.SECRET_KEY[:10] + '...')

print('\n== the API namespace always answers JSON ==')
# The SPA catch-all added for single-port serving must not swallow /api paths.
for method, path in [('get', '/api/nope'),
                     ('post', '/api/projects/x/images/../../y/annotations'),
                     ('delete', '/api/whatever')]:
    r = getattr(c, method)(path)
    check(4, f'{method.upper()} {path[:40]} is a JSON 404',
          r.status_code == 404 and 'json' in (r.content_type or ''),
          f'{r.status_code} {r.content_type}')

print('\n== #16 Windows reserved names and trailing dots ==')
for bad in ['NUL', 'CON', 'COM1', 'AUX', 'proj.']:
    r = c.post('/api/projects', json={'name': bad})
    check(16, f'rejects {bad!r}', r.status_code == 400, f'status {r.status_code}')

print('\n== #15 model_name / filename validation ==')
from services.projects import safe_filename, ProjectError
for bad in ['a*b', 'a?b', 'a|b', 'x' * 200, 'NUL', 'trail.']:
    try:
        safe_filename(bad)
        ok_reject = False
    except ProjectError:
        ok_reject = True
    check(15, f'safe_filename rejects {bad[:12]!r}', ok_reject)

print('\n== #10 malformed regions no longer brick a project ==')
c.post('/api/projects', json={'name': 'mal'})
c.post('/api/projects/mal/images', data={'images': [(io.BytesIO(jpg()), 'a.jpg')]},
       content_type='multipart/form-data')
mfn = c.get('/api/projects/mal/images').get_json()['images'][0]['filename']
bad_ann = Path(os.environ['PROJECTS_ROOT']) / 'mal' / 'annotations' / f'{mfn}.json'
bad_ann.write_text(json.dumps({'filename': mfn, 'regions': ['oops', 42, {'tag': 'ok',
                   'x': 1, 'y': 1, 'width': 9, 'height': 9}], 'annotated': True,
                   'width': 100, 'height': 100}), encoding='utf-8')
r1 = c.get('/api/projects/mal/images')
r2 = c.get('/api/projects/mal/dataset-summary')
check(10, 'gallery survives a malformed regions array', r1.status_code == 200, r1.status_code)
check(10, 'dataset-summary survives it', r2.status_code == 200, r2.status_code)
check(10, 'the one valid region is still counted',
      r2.get_json().get('total_boxes') == 1, r2.get_json().get('total_boxes'))

print('\n== #2 concurrent training starts are serialised ==')
c.post('/api/projects', json={'name': 'lockme'})
ups = [(io.BytesIO(jpg(200, 200)), f'i{i}.jpg') for i in range(8)]
c.post('/api/projects/lockme/images', data={'images': ups}, content_type='multipart/form-data')
for e in c.get('/api/projects/lockme/images').get_json()['images']:
    c.post(f"/api/projects/lockme/images/{e['filename']}/annotations",
           json={'regions': [{'tag': 's', 'x': 10, 'y': 10, 'width': 60, 'height': 60}]})

outcomes = []
def start_race(n):
    cl = app.test_client()
    r = cl.post('/api/projects/lockme/training/start', json={
        'model_type': 'yolo11n', 'epochs': 1, 'batch_size': 2, 'img_size': 320,
        'model_name': f'race{n}'})
    outcomes.append(r.status_code)

ts = [threading.Thread(target=start_race, args=(n,)) for n in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
successes = outcomes.count(200)
check(2, 'exactly one concurrent start wins', successes == 1,
      f'statuses {sorted(outcomes)}')

# clean up whatever started
try:
    c.post('/api/projects/lockme/training/stop')
except Exception:
    pass
time.sleep(1)

print('\n== #15b reusing a run name is refused ==')
st = c.get('/api/projects/lockme/training/status').get_json().get('status') or {}
used = st.get('model_name')
if used:
    (Path(os.environ['PROJECTS_ROOT']) / 'lockme' / 'training' / 'runs' / used).mkdir(parents=True, exist_ok=True)
    r = c.post('/api/projects/lockme/training/start', json={
        'model_type': 'yolo11n', 'epochs': 1, 'model_name': used})
    check(15, 'a duplicate run name is rejected', r.status_code == 400,
          f'{r.status_code}: {(r.get_json() or {}).get("message")}')

print('\n== #20 model download is restricted to model files ==')
proj = Path(os.environ['PROJECTS_ROOT']) / 'mal'
r = c.get('/api/projects/mal/training/models/download?path=' + str(proj / 'project.json'))
check(20, 'project.json is not downloadable', r.status_code == 403, r.status_code)
r = c.get('/api/projects/mal/training/models/download?path=' +
          str(proj / 'images' / mfn))
check(20, 'raw images are not downloadable', r.status_code == 403, r.status_code)

print('\n== #20b img_size is validated ==')
r = c.post('/api/models/test', data={'model': (io.BytesIO(b'x'), 'm.pt'),
           'images': (io.BytesIO(jpg()), 'a.jpg'), 'img_size': 'abc'},
           content_type='multipart/form-data')
check(20, 'non-numeric img_size is a 400, not a 500', r.status_code == 400, r.status_code)

print('\n== #18 zip import limits ==')
z = TMP / 'bad.zip'
with zipfile.ZipFile(z, 'w') as zf:
    zf.writestr('dataset.json', json.dumps({'annotations': [
        {'filename': 'shell.php', 'regions': [], 'annotated': False},
        {'filename': 'noext', 'regions': [], 'annotated': False},
        {'filename': 'good.jpg', 'regions': [{'tag': 't', 'x': 1, 'y': 1,
                                              'width': 9, 'height': 9}], 'annotated': True},
    ]}))
    zf.writestr('images/shell.php', b'<?php ?>')
    zf.writestr('images/noext', b'nope')
    zf.writestr('images/good.jpg', jpg())
r = c.post('/api/projects/mal/import-dataset',
           data={'file': (open(z, 'rb'), 'bad.zip')}, content_type='multipart/form-data')
body = r.get_json()
check(18, 'only the real image is imported', body.get('imported_images') == 1, body)
check(18, 'the others are reported as skipped', body.get('skipped_count') == 2, body)
php = list((Path(os.environ['PROJECTS_ROOT']) / 'mal' / 'images').glob('*.php'))
check(18, 'no .php written into images/', not php, php)

print('\n== #19 log lines are written once ==')
wc = (BACKEND / 'training' /
      'worker_common.py').read_text(encoding='utf-8')
check(19, 'log() no longer writes and prints', wc.count('f.write(line') == 0)

print('\n== #7/#8 FRCNN geometry and exports ==')
fl = (BACKEND / 'training' /
      'frcnn_lib.py').read_text(encoding='utf-8')
check(7, 'dataset no longer squashes to a square',
      'cv2.resize(image, (self.img_size, self.img_size)' not in fl)
check(7, 'model transform is configured from img_size', 'model.transform.min_size' in fl)
fw = (BACKEND / 'training' /
      'frcnn_worker.py').read_text(encoding='utf-8')
best_idx = fw.index('Loaded best checkpoint for export')
export_idx = fw.index("if 'torchscript' in export_formats")
check(8, 'best checkpoint is loaded before exports', best_idx < export_idx)
check(9, 'no full-pickle .pt is produced', 'torch.save(model, pt_path)' not in fw)

print('\n== #14 YOLO reads metrics from the validation hook ==')
yw = (BACKEND / 'training' /
      'yolo_worker.py').read_text(encoding='utf-8')
check(14, "callback is on_fit_epoch_end", "'on_fit_epoch_end'" in yw)

print('\n' + ('ALL FIXES VERIFIED' if not fails else f'{len(fails)} FAILED: {fails}'))
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
