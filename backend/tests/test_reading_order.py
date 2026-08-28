"""
Detections in the order a person reads them.

A detector returns boxes in whatever order it found them, which for most models
is by confidence. That is the wrong order for anything being read rather than
counted: a display showing 250 comes back as 0, 2, 5 or 5, 0, 2 depending on
which digit the model was surest about, and the number is lost. Every path that
returns detections has to put them back in reading order before anyone sees
them.

    python backend/tests/test_reading_order.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_order_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

from services import inference                          # noqa: E402

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


def box(label, x, y, w=60, h=100, score=0.5):
    return {'label_name': label, 'label_id': 0, 'score': score,
            'box': [x, y, x + w, y + h]}


print('== one line, found out of order ==')
# The order a detector actually emits: most confident first.
found = [box('0', 300, 100, score=0.95),
         box('2', 100, 100, score=0.71),
         box('5', 200, 105, score=0.83)]
check('the raw order is not the reading',
      ''.join(d['label_name'] for d in found) == '025')
check('sorted left to right', inference.reading_of(found) == '250',
      inference.reading_of(found))

ordered = inference.sort_reading_order(found)
check('every detection is kept', len(ordered) == 3, len(ordered))
check('each carries its position', [d['position'] for d in ordered] == [0, 1, 2],
      [d.get('position') for d in ordered])
check('all on one line', {d['line'] for d in ordered} == {0},
      [d.get('line') for d in ordered])

print('\n== two lines ==')
two = found + [box('A', 200, 300), box('B', 100, 305)]
check('lines are read top to bottom, each left to right',
      inference.reading_of(two) == '250\nBA', repr(inference.reading_of(two)))

print('\n== a digit sitting slightly high or low stays on its line ==')
# Real boxes are never perfectly aligned; a tolerance that is too tight splits
# one row into several and the reading comes out scrambled.
wobbly = [box('1', 100, 100), box('2', 170, 112), box('3', 240, 92),
          box('4', 310, 105)]
check('a 20px wobble does not split the line',
      inference.reading_of(wobbly) == '1234', inference.reading_of(wobbly))

print('\n== a genuinely lower row is a separate line ==')
stacked = [box('1', 100, 100), box('2', 100, 260)]
check('boxes that do not overlap vertically are two lines',
      inference.reading_of(stacked) == '1\n2', repr(inference.reading_of(stacked)))

print('\n== boxes of different sizes on the same line ==')
# A large digit beside a small one still shares most of the smaller one's
# height, which is what the overlap is measured against.
mixed = [box('7', 100, 100, w=40, h=60), box('8', 160, 80, w=80, h=110)]
check('a small box beside a tall one reads as one line',
      inference.reading_of(mixed) == '78', repr(inference.reading_of(mixed)))

print('\n== nothing found ==')
check('an empty list gives an empty reading', inference.reading_of([]) == '')
check('and sorts to an empty list', inference.sort_reading_order([]) == [])
check('None is handled', inference.reading_of(None) == '')

print('\n== the original detections are not mutated ==')
before = [dict(d) for d in found]
inference.sort_reading_order(found)
check('sorting leaves the caller\'s list alone', found == before)

print('\n== reading order survives a round trip through the API ==')
import io                                               # noqa: E402
import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
from app import create_app                              # noqa: E402

app = create_app()
c = app.test_client()

# Three digits drawn left to right; whatever a model makes of them, what comes
# back must be ordered by position rather than by score.
frame = np.full((240, 480, 3), 38, np.uint8)
for i, digit in enumerate('250'):
    cv2.putText(frame, digit, (60 + i * 130, 170), cv2.FONT_HERSHEY_SIMPLEX,
                4.0, (240, 248, 255), 10, cv2.LINE_AA)
encoded = cv2.imencode('.jpg', frame)[1].tobytes()

c.post('/api/projects', json={'name': 'order'})
weights = REPO / 'data' / 'weights' / 'yolo11n.pt'
if weights.is_file():
    r = c.post('/api/models/test',
               data={'images': (io.BytesIO(encoded), 'f.jpg'),
                     'model': (io.BytesIO(weights.read_bytes()), 'yolo11n.pt'),
                     'score_threshold': '0.05'},
               content_type='multipart/form-data')
    body = r.get_json()
    check('the endpoint answered', r.status_code == 200, body)
    result = (body.get('results') or [{}])[0]
    check('a reading field is present', 'reading' in result, sorted(result))
    detections = result.get('detections', [])
    if detections:
        xs = [d['box'][0] for d in detections if d.get('line') == 0]
        check('detections on a line come back sorted by x',
              xs == sorted(xs), xs)
        check('each detection knows its line and position',
              all('line' in d and 'position' in d for d in detections),
              detections[0])
    else:
        print('    (the stock model found nothing in a synthetic frame; '
              'the ordering itself is covered above)')
else:
    print('    (no pretrained weights on this checkout; skipped)')

print('\n' + ('READING ORDER OK' if not fails else f'{len(fails)} FAILED: {fails}'))
import shutil                                           # noqa: E402
if not fails:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
