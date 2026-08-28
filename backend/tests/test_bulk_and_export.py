"""
Removing images in bulk, and getting a video analysis out as CSV.

Both exist because the tool could produce far more than it could hand back.
Filters write a hundred images in seconds and deleting them was one request per
file; a video analysis knows what it read at every instant and the only way to
see it was to watch the boxes go past.

    python backend/tests/test_bulk_and_export.py
"""
import csv
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_bulk_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
from app import create_app                              # noqa: E402
from services import videojob                           # noqa: E402

app = create_app()
c = app.test_client()

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


print('== a project with photographs and generated copies ==')
c.post('/api/projects', json={'name': 'bulk'})
rng = np.random.default_rng(6)
uploads = []
for i in range(6):
    img = np.full((256, 256, 3), 36, np.uint8)
    img = cv2.add(img, rng.integers(0, 20, (256, 256, 3), dtype=np.uint8))
    cv2.rectangle(img, (70, 70), (170, 170), (238, 244, 250), -1)
    uploads.append((io.BytesIO(cv2.imencode('.jpg', img)[1].tobytes()), f'{i}.jpg'))

r = c.post('/api/projects/bulk/images', data={'images': uploads},
           content_type='multipart/form-data')
names = [x['filename'] for x in r.get_json()['imported']]
for name in names:
    c.post(f'/api/projects/bulk/images/{name}/annotations',
           json={'regions': [{'tag': 'block', 'x': 68, 'y': 68,
                              'width': 104, 'height': 104}]})

# How many copies survive is up to the survival guard, which drops any whose
# filter hid the annotated object. Taking the count from the response rather
# than assuming it keeps this test about deleting rather than about filtering.
made = c.post('/api/projects/bulk/augment-color',
              json={'tones': ['clahe', 'tophat'], 'variants_per_tone': 1}).get_json()
generated = made['created_count']
print(f"    {generated} copies kept, {made.get('dropped_by_tone') or 'none'} dropped")
check('some copies were generated', generated > 0, made)

summary = c.get('/api/projects/bulk/dataset-summary').get_json()
check('the gallery holds the photographs and the copies',
      summary['total_images'] == 6 + generated, summary['total_images'])

print('\n== deleting only what the filters made ==')
r = c.post('/api/projects/bulk/images/delete', json={'only_generated': True})
body = r.get_json()
check('the request succeeded', r.status_code == 200, body)
check('every generated copy was removed', body['deleted_count'] == generated,
      (body['deleted_count'], generated))
check('nothing failed', not body['failed'], body['failed'])

after = c.get('/api/projects/bulk/dataset-summary').get_json()
check('the photographs are untouched', after['total_images'] == 6,
      after['total_images'])
check('and they are still annotated', after['annotated_images'] == 6,
      after['annotated_images'])
check('the counts in the response match', body['total_images'] == 6,
      body['total_images'])

listed = c.get('/api/projects/bulk/images').get_json()['images']
check('no generated image is left in the gallery',
      not any(i['augmented'] for i in listed),
      [i['filename'] for i in listed if i['augmented']][:2])

print('\n== deleting a named list ==')
r = c.post('/api/projects/bulk/images/delete',
           json={'filenames': names[:2]})
body = r.get_json()
check('two were removed', body['deleted_count'] == 2, body['deleted_count'])
check('four remain',
      c.get('/api/projects/bulk/dataset-summary').get_json()['total_images'] == 4)

print('\n== a name that does not exist, and one that is not allowed ==')
r = c.post('/api/projects/bulk/images/delete',
           json={'filenames': ['nope.jpg', '../../etc/passwd', names[2]]})
body = r.get_json()
check('the request still succeeds', r.status_code == 200, body)
check('the real file was removed', names[2] in body['deleted'], body['deleted'])
check('a traversal attempt is not acted on',
      not any('passwd' in n for n in body['deleted']), body['deleted'])

print('\n== an empty request changes nothing ==')
before = c.get('/api/projects/bulk/dataset-summary').get_json()['total_images']
r = c.post('/api/projects/bulk/images/delete', json={})
check('nothing is deleted', r.get_json()['deleted_count'] == 0, r.get_json())
check('the count is unchanged',
      c.get('/api/projects/bulk/dataset-summary').get_json()['total_images'] == before)

print('\n== a video analysis as CSV ==')
# The job is built directly rather than by running a model: what is being
# checked is the shape of the export, not the detecting.
job_id = 'csvtest0001'
videojob._jobs[job_id] = {
    'id': job_id, 'status': 'completed', 'message': '', 'filename': 'line.mp4',
    'model_name': 'best.pt', 'model_path': '', 'source_path': '',
    'score_threshold': 0.25, 'label_names': ['2', '5', '0'], 'img_size': 640,
    'sample_fps': 5, 'fps': 10, 'width': 640, 'height': 480, 'duration_s': 2.0,
    'frames_total': 3, 'frames_done': 3, 'detection_count': 4,
    'started_at': '', 'finished_at': '', 'elapsed_s': 1.0, 'cancel': False,
    'frames': [
        {'time_s': 0.0, 'frame': 0, 'reading': '250', 'detections': [
            {'label_id': 0, 'label_name': '2', 'score': 0.91,
             'box': [10, 20, 60, 120], 'line': 0, 'position': 0},
            {'label_id': 1, 'label_name': '5', 'score': 0.88,
             'box': [70, 20, 120, 120], 'line': 0, 'position': 1},
            {'label_id': 2, 'label_name': '0', 'score': 0.85,
             'box': [130, 20, 180, 120], 'line': 0, 'position': 2},
        ]},
        {'time_s': 0.2, 'frame': 2, 'reading': '', 'detections': []},
        {'time_s': 0.4, 'frame': 4, 'reading': '2', 'detections': [
            {'label_id': 0, 'label_name': '2', 'score': 0.44,
             'box': [12, 22, 62, 122], 'line': 0, 'position': 0},
        ]},
    ],
}

r = c.get(f'/api/models/video/{job_id}/csv')
check('the export succeeded', r.status_code == 200, r.status_code)
check('it is offered as a file',
      'attachment' in r.headers.get('Content-Disposition', ''),
      r.headers.get('Content-Disposition'))
check('named after the clip',
      'line_detections.csv' in r.headers.get('Content-Disposition', ''),
      r.headers.get('Content-Disposition'))
check('it is CSV', r.mimetype == 'text/csv', r.mimetype)

rows = list(csv.reader(io.StringIO(r.get_data(as_text=True))))
header, body_rows = rows[0], rows[1:]
check('the header names every column',
      header == ['time_s', 'frame', 'reading', 'label', 'score',
                 'x1', 'y1', 'x2', 'y2', 'line', 'position'], header)
check('one row per detection, plus the empty frame',
      len(body_rows) == 5, len(body_rows))
check('the first row carries the whole reading',
      body_rows[0][2] == '250' and body_rows[0][3] == '2', body_rows[0])
check('detections stay in reading order',
      [row[3] for row in body_rows[:3]] == ['2', '5', '0'],
      [row[3] for row in body_rows[:3]])
check('a frame where nothing was found is still a row',
      body_rows[3][0] == '0.2' and body_rows[3][3] == '', body_rows[3])
check('the box coordinates come through',
      body_rows[0][5:9] == ['10', '20', '60', '120'], body_rows[0][5:9])

print('\n== a job that is not finished, and one that does not exist ==')
videojob._jobs['running0001'] = {**videojob._jobs[job_id], 'id': 'running0001',
                                 'status': 'running'}
r = c.get('/api/models/video/running0001/csv')
check('an unfinished analysis is refused', r.status_code == 400, r.status_code)
r = c.get('/api/models/video/nosuchjob99/csv')
check('an unknown job is a 404', r.status_code == 404, r.status_code)

print('\n' + ('BULK AND EXPORT OK' if not fails else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
