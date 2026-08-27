"""
Live detection: one frame at a time, and a whole video.

Both paths exist because the still-image endpoint is the wrong shape for a
stream. It reloads the weights per call and returns an annotated JPEG, which is
fine for a handful of pictures and hopeless several times a second. These
return coordinates and keep the model in memory, and the numbers below check
that both of those actually happen rather than being merely intended.

    python backend/tests/test_video_webcam.py
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
TMP = Path(tempfile.mkdtemp(prefix='vt_live_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
from app import create_app                              # noqa: E402
from services import modelcache                         # noqa: E402

app = create_app()
c = app.test_client()

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


def frame_with_square(width=416, height=352, x=150, y=120, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 38, np.uint8)
    img = cv2.add(img, rng.integers(0, 22, (height, width, 3), dtype=np.uint8))
    cv2.rectangle(img, (x, y), (x + 76, y + 76), (232, 240, 250), -1)
    return img


PROJECT = 'live'
c.post('/api/projects', json={'name': PROJECT})

print('== train something to detect with ==')
uploads, boxes = [], []
rng = np.random.default_rng(11)
for i in range(40):
    x, y = int(rng.integers(40, 300)), int(rng.integers(40, 220))
    img = frame_with_square(x=x, y=y, seed=i)
    uploads.append((io.BytesIO(cv2.imencode('.jpg', img)[1].tobytes()), f'f{i:02d}.jpg'))
    boxes.append((x - 3, y - 3, 82, 82))

r = c.post(f'/api/projects/{PROJECT}/images', data={'images': uploads},
           content_type='multipart/form-data')
names = [x['filename'] for x in r.get_json()['imported']]
for name, (x, y, w, h) in zip(names, boxes):
    c.post(f'/api/projects/{PROJECT}/images/{name}/annotations',
           json={'regions': [{'tag': 'block', 'x': x, 'y': y, 'width': w, 'height': h}]})

print('    training (a few minutes on a CPU)...')
c.post(f'/api/projects/{PROJECT}/training/start', json={
    'model_type': 'yolo11n', 'epochs': 50, 'batch_size': 8,
    'img_size': 192, 'export_formats': ['pt', 'onnx'], 'model_name': 'live_run'})

deadline = time.time() + 900
state = None
while time.time() < deadline:
    state = ((c.get(f'/api/projects/{PROJECT}/training/status').get_json()
              ['status']) or {}).get('status')
    if state in ('completed', 'failed', 'stopped', 'error'):
        break
    time.sleep(4)
check('a model was trained to test with', state == 'completed', state)

entries = c.get(f'/api/projects/{PROJECT}/training/models').get_json()['models']
weights = entries[0]['path'] if entries else None
check('the weights are on disk', bool(weights) and Path(weights).is_file(), weights)

if not weights:
    print('\nno model to test with; stopping')
    sys.exit(1)

probe = frame_with_square(x=150, y=120, seed=99)
encoded = cv2.imencode('.jpg', probe)[1].tobytes()

print('\n== one frame, coordinates only ==')
r = c.post('/api/models/detect',
           data={'frame': (io.BytesIO(encoded), 'f.jpg'),
                 'model_path': weights, 'score_threshold': '0.05',
                 'img_size': '192'},
           content_type='multipart/form-data')
body = r.get_json()
check('the frame was accepted', r.status_code == 200, body)
check('it found the object', bool(body.get('detections')), body)
check('no annotated image was sent back', 'annotated_image' not in str(body)[:2000])
check('the frame size came back', body.get('width') == 416 and body.get('height') == 352,
      (body.get('width'), body.get('height')))
check('the class names came back', bool(body.get('label_names')), body.get('label_names'))
if body.get('detections'):
    best = max(body['detections'], key=lambda d: d['score'])
    cx = (best['box'][0] + best['box'][2]) / 2
    cy = (best['box'][1] + best['box'][3]) / 2
    check('the box is on the object',
          150 <= cx <= 226 and 120 <= cy <= 196, (cx, cy))

print('\n== the model stays in memory between frames ==')
# This is the whole reason the endpoint exists. Loading a small YOLO costs
# about three times what predicting a frame does, so a feed that reloads is
# capped near 6 fps where a cached one reaches 25.
modelcache.clear()
started = time.time()
c.post('/api/models/detect',
       data={'frame': (io.BytesIO(encoded), 'f.jpg'), 'model_path': weights,
             'img_size': '192'},
       content_type='multipart/form-data')
cold = time.time() - started

times = []
for _ in range(5):
    started = time.time()
    c.post('/api/models/detect',
           data={'frame': (io.BytesIO(encoded), 'f.jpg'), 'model_path': weights,
                 'img_size': '192'},
           content_type='multipart/form-data')
    times.append(time.time() - started)
warm = sum(times) / len(times)
print(f'    first frame {cold * 1000:.0f} ms, then {warm * 1000:.0f} ms each '
      f'({1 / warm:.1f} fps)')
check('later frames are faster than the first', warm < cold,
      f'{warm * 1000:.0f}ms vs {cold * 1000:.0f}ms')
check('the cache is holding the model', len(modelcache.describe()) == 1,
      modelcache.describe())

print('\n== retraining into the same filename is noticed ==')
# The cache keys on size and modification time as well as the path, so a model
# replaced in place must not keep serving the previous run's predictions.
before = len(modelcache.describe())
Path(weights).touch()
c.post('/api/models/detect',
       data={'frame': (io.BytesIO(encoded), 'f.jpg'), 'model_path': weights,
             'img_size': '192'},
       content_type='multipart/form-data')
check('a changed file is reloaded rather than served from cache',
      len(modelcache.describe()) == before + 1, modelcache.describe())

print('\n== a model is run at the size it was exported for ==')
# An ONNX export has its resolution compiled into the graph. The test screen
# sends 640 unless told otherwise, and a model exported at 320 then failed with
# "Got invalid dimensions for input: images ... Got: 640 Expected: 320" from
# onnxruntime, which reached the browser as an opaque 500.
# The listing is newest first, so which format lands at index 0 depends on the
# export order. Pick by suffix rather than by position.
torch_weights = next((e['path'] for e in entries if e['path'].endswith('.pt')), None)
check('the run produced a .pt', bool(torch_weights), [e['path'] for e in entries])

native, fixed = modelcache.native_input_size(torch_weights)
check('the .pt reports the size it was trained at', native == 192, (native, fixed))
check('and does not claim that size is the only one allowed', fixed is False)

onnx_exports = list(Path(weights).parent.glob('*.onnx'))
if onnx_exports:
    onnx_size, onnx_fixed = modelcache.native_input_size(onnx_exports[0])
    check('an ONNX export reports a fixed size', bool(onnx_size) and onnx_fixed,
          (onnx_size, onnx_fixed))
    r = c.post('/api/models/detect',
               data={'frame': (io.BytesIO(encoded), 'f.jpg'),
                     'model_path': str(onnx_exports[0]), 'img_size': '640'},
               content_type='multipart/form-data')
    check('asking an ONNX export for the wrong size does not 500',
          r.status_code == 200, (r.status_code, r.get_json()))
else:
    print('    (no ONNX export in this run; the .pt checks above still apply)')

r = c.post('/api/models/detect',
           data={'frame': (io.BytesIO(encoded), 'f.jpg'),
                 'model_path': torch_weights, 'img_size': '640'},
           content_type='multipart/form-data')
check('a 192-trained model still detects when the screen asks for 640',
      r.status_code == 200 and bool(r.get_json().get('detections')),
      (r.status_code, str(r.get_json())[:160]))

print('\n== a model outside the projects tree is still refused ==')
r = c.post('/api/models/detect',
           data={'frame': (io.BytesIO(encoded), 'f.jpg'),
                 'model_path': str(REPO / 'backend' / 'app.py')},
           content_type='multipart/form-data')
check('the path check applies to the live endpoint too',
      r.status_code in (400, 403), r.status_code)

print('\n== an unreadable frame is reported, not crashed on ==')
r = c.post('/api/models/detect',
           data={'frame': (io.BytesIO(b'not an image'), 'f.jpg'),
                 'model_path': weights},
           content_type='multipart/form-data')
check('a corrupt frame gives a clear 400', r.status_code == 400,
      (r.status_code, r.get_json()))

print('\n== a whole video ==')
video_path = TMP / 'clip.mp4'
writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'),
                         10.0, (416, 352))
check('a test clip could be written', writer.isOpened())
for i in range(40):                       # 4 seconds at 10 fps, object moving
    writer.write(frame_with_square(x=60 + i * 5, y=120, seed=i))
writer.release()

r = c.post('/api/models/video',
           data={'video': (io.BytesIO(video_path.read_bytes()), 'clip.mp4'),
                 'model_path': weights, 'score_threshold': '0.05',
                 'img_size': '192', 'sample_fps': '5'},
           content_type='multipart/form-data')
body = r.get_json()
check('the video was accepted', r.status_code == 200, body)
job = (body or {}).get('job') or {}
check('a job id came back', bool(job.get('id')), job)
check('the clip was probed', job.get('fps') and job.get('width') == 416, job)

job_id = job.get('id')
deadline = time.time() + 300
while job_id and time.time() < deadline:
    job = c.get(f'/api/models/video/{job_id}').get_json()['job']
    if job['status'] != 'running':
        break
    time.sleep(1)

check('the video job finished', job.get('status') == 'completed', job.get('message'))
print(f"    {job.get('frames_total')} frames sampled, "
      f"{job.get('detection_count')} detections, {job.get('elapsed_s')}s")
check('it sampled at the rate asked for',
      job.get('frames_total') in (20, 21), job.get('frames_total'))
check('it found the object in the clip', job.get('detection_count', 0) > 0,
      job.get('message'))
check('every sample carries a timestamp',
      all('time_s' in f for f in job.get('frames', [])), (job.get('frames') or [])[:1])
check('the timestamps run in order',
      [f['time_s'] for f in job.get('frames', [])] ==
      sorted(f['time_s'] for f in job.get('frames', [])))
check('no video file is sent back', 'video' not in job and 'download_url' not in job)

moving = [f for f in job.get('frames', []) if f['detections']]
if len(moving) >= 2:
    first_x = moving[0]['detections'][0]['box'][0]
    last_x = moving[-1]['detections'][0]['box'][0]
    check('the box follows the object across the clip', last_x > first_x,
          (first_x, last_x))

print('\n== a video the decoder cannot open is reported clearly ==')
r = c.post('/api/models/video',
           data={'video': (io.BytesIO(b'not a video at all'), 'broken.mp4'),
                 'model_path': weights},
           content_type='multipart/form-data')
check('an unopenable video gives a 400 with an explanation',
      r.status_code == 400 and 'video' in (r.get_json() or {}).get('message', '').lower(),
      (r.status_code, r.get_json()))

print('\n== a file that is not a video at all is refused on its extension ==')
r = c.post('/api/models/video',
           data={'video': (io.BytesIO(b'x'), 'notes.txt'), 'model_path': weights},
           content_type='multipart/form-data')
check('a non-video extension is refused', r.status_code == 400, r.status_code)

print('\n== asking about a job that does not exist ==')
r = c.get('/api/models/video/deadbeef1234')
check('an unknown job id is a 404', r.status_code == 404, r.status_code)

print('\n' + ('LIVE DETECTION OK' if not fails else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    modelcache.clear()
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
