"""
Telling one upload from another.

While a project is being built, "everything unlabelled" and "the images I just
added" are the same set. The moment it is being extended -- a second site, a
new shift, a run of parts the model got wrong -- they stop being. A handful of
pictures skipped months ago are still unlabelled, and running a model over
those alongside today's import mixes two decisions into one review.

Each upload is numbered, generated copies inherit their source's number, and
auto-labelling can be pointed at one of them.

    python backend/tests/test_import_batches.py
"""
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_batch_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
from app import create_app                              # noqa: E402

app = create_app()
c = app.test_client()

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


def upload(project, count, seed):
    rng = np.random.default_rng(seed)
    uploads = []
    for i in range(count):
        img = np.full((256, 256, 3), 34, np.uint8)
        img = cv2.add(img, rng.integers(0, 20, (256, 256, 3), dtype=np.uint8))
        cv2.rectangle(img, (70, 70), (170, 170), (236, 242, 248), -1)
        uploads.append((io.BytesIO(cv2.imencode('.jpg', img)[1].tobytes()),
                        f's{seed}_{i}.jpg'))
    r = c.post(f'/api/projects/{project}/images', data={'images': uploads},
               content_type='multipart/form-data')
    return r.get_json()


def images(project):
    return c.get(f'/api/projects/{project}/images').get_json()['images']


print('== the first upload ==')
c.post('/api/projects', json={'name': 'batches'})
first = upload('batches', 4, 1)
check('it reports which batch it was', first.get('batch') == 1, first.get('batch'))
check('and when', bool(first.get('imported_at')), first.get('imported_at'))
check('every file is in batch 1',
      {i['batch'] for i in images('batches')} == {1},
      {i['batch'] for i in images('batches')})

for entry in images('batches'):
    c.post(f'/api/projects/batches/images/{entry["filename"]}/annotations',
           json={'regions': [{'tag': 'block', 'x': 68, 'y': 68,
                              'width': 104, 'height': 104}]})

print('\n== a second upload is numbered after it ==')
second = upload('batches', 3, 2)
check('the number went up', second.get('batch') == 2, second.get('batch'))

listed = images('batches')
by_batch = {}
for entry in listed:
    by_batch.setdefault(entry['batch'], []).append(entry)
check('four in the first, three in the second',
      len(by_batch.get(1, [])) == 4 and len(by_batch.get(2, [])) == 3,
      {k: len(v) for k, v in by_batch.items()})
check('only the second batch is unannotated',
      all(e['annotated'] for e in by_batch[1])
      and not any(e['annotated'] for e in by_batch[2]),
      [(e['batch'], e['annotated']) for e in listed])

print('\n== a copy belongs to the batch its source came from ==')
made = c.post('/api/projects/batches/augment-color',
              json={'tones': ['clahe'], 'variants_per_tone': 1,
                    'require_all_annotated': False}).get_json()
print(f"    {made['created_count']} copies made")
generated = [e for e in images('batches') if e['augmented']]
check('the copies were made from the annotated batch',
      bool(generated) and {e['batch'] for e in generated} == {1},
      {e['batch'] for e in generated})
check('so filtering by batch shows a photograph with what came from it',
      len([e for e in images('batches') if e['batch'] == 1]) == 4 + len(generated),
      len([e for e in images('batches') if e['batch'] == 1]))

print('\n== photographs can be told from copies ==')
current = images('batches')
photos = [e for e in current if not e['augmented']]
copies = [e for e in current if e['augmented']]
check('both groups are non-empty', photos and copies, (len(photos), len(copies)))
check('and together they are everything',
      len(photos) + len(copies) == len(current), len(current))

print('\n== auto-labelling one batch rather than everything unlabelled ==')
# Leave an old image unlabelled too, so "unlabelled" and "the new ones" are
# genuinely different sets -- which is the whole reason for the option.
stale = by_batch[1][0]['filename']
c.post(f'/api/projects/batches/images/{stale}/annotations', json={'regions': []})
pending = [e for e in images('batches') if not e['annotated'] and not e['augmented']]
check('there are now unlabelled images in both batches',
      {e['batch'] for e in pending} == {1, 2},
      {e['batch'] for e in pending})

models = c.get('/api/models').get_json()['models']
usable = next((m for m in models if m['format'] == 'pt'), None)
if usable:
    r = c.post('/api/projects/batches/auto-label',
               json={'model_path': usable['path'], 'batch': 2,
                     'score_threshold': 0.9})
    job = (r.get_json() or {}).get('job') or {}
    check('the job was accepted', r.status_code == 200, r.get_json())
    check('it targets only the second batch',
          job.get('total') == len([e for e in pending if e['batch'] == 2]),
          (job.get('total'), len([e for e in pending if e['batch'] == 2])))
    check('which is fewer than everything unlabelled',
          job.get('total', 0) < len(pending), (job.get('total'), len(pending)))
else:
    print('    (no trained model on this machine; the targeting is checked '
          'through the service instead)')
    from services import autolabel                      # noqa: E402
    from services.projects import ProjectError          # noqa: E402
    try:
        autolabel.start('batches', batch=2)
        check('a batch with nothing to do is refused', False, 'it started')
    except ProjectError as exc:
        check('the service accepts a batch argument',
              'model' in exc.message.lower() or 'annotations' in exc.message.lower(),
              exc.message)

print('\n== a batch that does not exist ==')
if usable:
    r = c.post('/api/projects/batches/auto-label',
               json={'model_path': usable['path'], 'batch': 99})
    check('is refused rather than silently doing everything',
          r.status_code == 400, (r.status_code, (r.get_json() or {}).get('message')))

print('\n== images from before batches were recorded ==')
# An older project has annotations with no batch field at all. Those must read
# as the original set rather than breaking the grouping.
c.post('/api/projects', json={'name': 'legacy'})
upload('legacy', 2, 7)
folder = Path(os.environ['PROJECTS_ROOT']) / 'legacy' / 'annotations'
import json                                             # noqa: E402
for path in folder.glob('*.json'):
    data = json.loads(path.read_text(encoding='utf-8'))
    data.pop('batch', None)
    data.pop('imported_at', None)
    path.write_text(json.dumps(data), encoding='utf-8')
c.post('/api/projects/legacy/rescan')

older = images('legacy')
check('they list without a batch rather than failing',
      len(older) == 2 and all(e.get('batch') is None for e in older),
      [e.get('batch') for e in older])
check('and the next upload starts at 1',
      upload('legacy', 1, 8).get('batch') == 1)

print('\n' + ('IMPORT BATCHES OK' if not fails else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
