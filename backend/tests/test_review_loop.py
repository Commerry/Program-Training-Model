"""
What a person changed about a prediction, recorded while both still exist.

A corrected picture is worth more than either half of it. The boxes kept say
the model was right, the ones moved say it was close, the deleted ones are
false positives and the drawn ones are objects it missed. That is the whole of
what there is to learn from, and it exists for exactly as long as it takes to
save the correction over the prediction.

A system that learns from its own predictions grows more certain and no more
correct -- it marks its own work. One that learns from corrections is being
told, by somebody who looked, where it was wrong. So the four kinds of change
are counted here, and counted as arithmetic rather than inferred, because the
whole value of the signal is that the model had no hand in producing it.

    python backend/tests/test_review_loop.py
"""
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_review_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
from app import create_app                              # noqa: E402
from services import projects, review                   # noqa: E402

app = create_app()
c = app.test_client()

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


def box(tag, x, y, w, h, score=None):
    region = {'tag': tag, 'x': float(x), 'y': float(y),
              'width': float(w), 'height': float(h)}
    if score is not None:
        region['score'] = score
    return region


print('== the four things a person can do to a predicted box ==')
predicted = [
    box('Good', 100, 100, 80, 80, 0.95),      # kept as it is
    box('CuffTear', 300, 100, 60, 60, 0.71),  # nudged into place
    box('DirtS', 100, 300, 50, 50, 0.55),     # was actually a DirtM
    box('Hole', 400, 400, 40, 40, 0.42),      # not there at all
]
saved = [
    box('Good', 100, 100, 80, 80),
    box('CuffTear', 315, 108, 60, 60),        # same object, moved
    box('DirtM', 100, 300, 50, 50),           # same box, different class
    box('Split', 500, 200, 70, 70),           # drawn by hand, missed entirely
]

result = review.compare(predicted, saved)
counts = result['counts']
check('a box left alone is kept', counts['kept'] == 1, counts)
check('a box nudged is moved', counts['moved'] == 1, counts)
check('a box renamed is relabelled', counts['relabelled'] == 1, counts)
check('a box removed is deleted', counts['deleted'] == 1, counts)
check('a box drawn by hand is added', counts['added'] == 1, counts)
check('agreement is the share the model got untouched',
      counts['agreement'] == 0.25, counts['agreement'])

print('\n== a relabelled box is a fact about the class the model chose ==')
summary = review.summarise([{'changes': result['changes']}])
by_tag = {row['tag']: row for row in summary['per_class']}
check('the mistake is filed under DirtS, not DirtM',
      by_tag.get('DirtS', {}).get('relabelled') == 1, by_tag.get('DirtS'))
check('and the missed object under the class it turned out to be',
      by_tag.get('Split', {}).get('added') == 1, by_tag.get('Split'))
check('a class the model never gets wrong has no correction rate',
      by_tag.get('Good', {}).get('correction_rate') == 0.0, by_tag.get('Good'))
check('and the worst class is listed first',
      summary['per_class'][0]['correction_rate'] >= summary['per_class'][-1]['correction_rate'],
      [(r['tag'], r['correction_rate']) for r in summary['per_class']])

print('\n== nothing to compare against ==')
check('a picture drawn from scratch is not a review',
      review.record_for({'regions': []}, saved) is None)
check('and a second save is not a second review',
      review.record_for({'auto_labelled': True, 'regions': predicted,
                         'review': {'counts': {}}}, saved) is None)

print('\n== through the application ==')
r = c.post('/api/projects', json={'name': 'gloves'})
check('the project is created', r.status_code in (200, 201), r.status_code)

frame = np.full((600, 800, 3), 30, np.uint8)
cv2.rectangle(frame, (100, 100), (180, 180), (220, 225, 235), -1)
encoded = cv2.imencode('.jpg', frame)[1].tobytes()
r = c.post('/api/projects/gloves/images',
           data={'images': [(io.BytesIO(encoded), 'a.jpg'),
                            (io.BytesIO(encoded), 'b.jpg')]},
           content_type='multipart/form-data')
check('the photographs upload', r.status_code == 200, r.status_code)

images = (c.get('/api/projects/gloves/images').get_json() or {}).get('images') or []
names = [i['filename'] for i in images]
check('both are there', len(names) == 2, names)

# Stand in for a labelling pass: write predictions the way one does.
with app.app_context():
    for index, filename in enumerate(names):
        stored = projects.read_annotation('gloves', filename) or {}
        stored.update({
            'filename': filename,
            'regions': predicted if index == 0 else [box('Good', 50, 50, 40, 40, 0.31)],
            'annotated': True,
            'width': 800, 'height': 600,
            'auto_labelled': True,
            'auto_label': {'model': 'imported/model.onnx', 'score_threshold': 0.3},
        })
        projects.write_annotation('gloves', filename, stored)
        projects._update_index_entry('gloves', filename, stored)

print('\n== the queue puts the doubtful picture first ==')
r = c.get('/api/projects/gloves/review/queue')
body = r.get_json() or {}
waiting = body.get('images') or []
check('both are waiting', body.get('waiting') == 2, body.get('waiting'))
check('the one it was least sure of comes first',
      bool(waiting) and waiting[0]['filename'] == names[1],
      [(w['filename'], w['informativeness']) for w in waiting])
check('and the reason is shown',
      bool(waiting) and waiting[0]['lowest_score'] == 0.31,
      waiting[0] if waiting else None)

print('\n== correcting one records what changed ==')
r = c.post(f'/api/projects/gloves/images/{names[0]}/annotations',
           json={'regions': saved})
check('the correction saves', r.status_code == 200, r.get_json())

with app.app_context():
    stored = projects.read_annotation('gloves', names[0]) or {}
check('a review was written', bool(stored.get('review')), list(stored))
recorded = (stored.get('review') or {}).get('counts') or {}
check('with the same counts as the comparison',
      {k: recorded.get(k) for k in ('kept', 'moved', 'relabelled', 'deleted', 'added')}
      == {'kept': 1, 'moved': 1, 'relabelled': 1, 'deleted': 1, 'added': 1},
      recorded)
check('and it remembers which model was being corrected',
      (stored.get('review') or {}).get('model') == 'imported/model.onnx',
      (stored.get('review') or {}).get('model'))
check('the corrected boxes are what is stored',
      len(stored.get('regions') or []) == 4, stored.get('regions'))

print('\n== and it leaves the queue ==')
body = c.get('/api/projects/gloves/review/queue').get_json() or {}
check('one left to check', body.get('waiting') == 1, body.get('waiting'))

print('\n== the summary is arithmetic, not opinion ==')
body = c.get('/api/projects/gloves/review/summary').get_json() or {}
totals = body.get('totals') or {}
check('one picture reviewed', totals.get('images') == 1, totals)
check('every kind of change counted', totals.get('relabelled') == 1, totals)
check('and one still pending', body.get('pending') == 1, body.get('pending'))
check('agreement is over what the model drew, not what was saved',
      body.get('agreement') == 0.25, body.get('agreement'))

print('\n== saving again does not double-count ==')
r = c.post(f'/api/projects/gloves/images/{names[0]}/annotations',
           json={'regions': saved})
body = c.get('/api/projects/gloves/review/summary').get_json() or {}
check('still one review', (body.get('totals') or {}).get('images') == 1,
      body.get('totals'))

print('\n' + ('REVIEW LOOP OK' if not fails else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
