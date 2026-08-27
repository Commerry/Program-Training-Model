"""
Concurrency stress: many writers hitting the same project at once.

Unlike the earlier edge-case script this checks HTTP status codes, so a 500
from a crashed handler is a failure rather than something the loop ignores.
"""
import io, os, sys, tempfile, shutil, json, threading, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix='vt_conc_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import numpy as np, cv2
from app import create_app

app = create_app()
app.config['PROPAGATE_EXCEPTIONS'] = False
c = app.test_client()
c.post('/api/projects', json={'name': 'race'})

uploads = []
for i in range(12):
    img = np.full((200, 200, 3), 40, np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (0, 200, 0), -1)
    ok, buf = cv2.imencode('.jpg', img)
    uploads.append((io.BytesIO(buf.tobytes()), f'i{i}.jpg'))
names = [x['filename'] for x in c.post('/api/projects/race/images',
         data={'images': uploads}, content_type='multipart/form-data'
         ).get_json()['imported']]

fails = []
def check(label, cond, detail=''):
    print(('  PASS ' if cond else '  FAIL ') + label + ('' if cond else f'  -> {detail}'))
    if not cond: fails.append(label)

print(f'== {len(names)} images, 8 threads x 25 saves each ==')
bad_status, exceptions = [], []
lock = threading.Lock()

def worker(tid):
    client = app.test_client()
    for i in range(25):
        target = names[(tid + i) % len(names)]
        try:
            r = client.post(f'/api/projects/race/images/{target}/annotations',
                            json={'regions': [{'tag': f't{tid}', 'x': 10, 'y': 10,
                                               'width': 40, 'height': 40}]})
            if r.status_code != 200:
                with lock:
                    bad_status.append((r.status_code, (r.get_json() or {}).get('message')))
        except Exception as exc:
            with lock:
                exceptions.append(repr(exc))

start = time.time()
threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
[t.start() for t in threads]
[t.join() for t in threads]
elapsed = time.time() - start

print(f'   200 saves in {elapsed:.1f}s ({200 / elapsed:.0f}/s)')
check('no exceptions', not exceptions, exceptions[:3])
check('every save returned 200', not bad_status, bad_status[:5])

print('\n== files are intact and parseable ==')
proj = Path(os.environ['PROJECTS_ROOT']) / 'race'
meta = json.loads((proj / 'project.json').read_text(encoding='utf-8'))
check('project.json is valid JSON', isinstance(meta.get('tags'), dict), meta.get('tags'))

broken = []
for f in (proj / 'annotations').glob('*.json'):
    try:
        json.loads(f.read_text(encoding='utf-8'))
    except Exception as e:
        broken.append((f.name, repr(e)))
check('every annotation file is valid JSON', not broken, broken[:3])

leftovers = list(proj.rglob('*.tmp'))
check('no temp files left behind', not leftovers, [p.name for p in leftovers][:5])

print('\n== concurrent reads while writing ==')
read_errors = []
stop_flag = threading.Event()

def reader():
    client = app.test_client()
    while not stop_flag.is_set():
        r = client.get('/api/projects/race/dataset-summary')
        if r.status_code != 200:
            read_errors.append((r.status_code, (r.get_json() or {}).get('message')))

def writer():
    client = app.test_client()
    for i in range(40):
        client.post(f'/api/projects/race/images/{names[i % len(names)]}/annotations',
                    json={'regions': [{'tag': 'x', 'x': 1, 'y': 1, 'width': 30, 'height': 30}]})

readers = [threading.Thread(target=reader) for _ in range(3)]
writers = [threading.Thread(target=writer) for _ in range(3)]
[t.start() for t in readers + writers]
[t.join() for t in writers]
stop_flag.set()
[t.join() for t in readers]
check('reads never fail while writes are in flight', not read_errors, read_errors[:3])

print('\n== concurrent project create/delete ==')
create_errors = []
def churn(n):
    client = app.test_client()
    for i in range(5):
        client.post('/api/projects', json={'name': f'p{n}_{i}'})
        r = client.get('/api/projects')
        if r.status_code != 200:
            create_errors.append(r.status_code)
        client.delete(f'/api/projects/p{n}_{i}')

ts = [threading.Thread(target=churn, args=(n,)) for n in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
check('listing stays healthy during create/delete churn', not create_errors, create_errors[:3])

print('\n' + ('CONCURRENCY OK' if not fails else f'{len(fails)} FAILED: {fails}'))
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
