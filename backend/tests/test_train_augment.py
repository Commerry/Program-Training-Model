"""
What a training run does to each image, and what it writes before starting.

Nothing in this application used to pass ultralytics any augmentation settings,
so it used its own defaults. Those mirror half of every epoch. For a project
whose classes are digits that is not a small inefficiency but a source of wrong
labels: a mirrored 2 is not a 2, and the label travelling with it still says it
is. The checks below fix that in place and make sure the opposite case -- a
project whose classes are objects, where mirroring is free extra data -- is not
broken in the process.

    python backend/tests/test_train_augment.py
"""
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_taug_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
from app import create_app                              # noqa: E402
from services import trainaug                           # noqa: E402

app = create_app()
c = app.test_client()

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


def make_project(name, tags, count=15):
    """A project with `count` annotated images, cycling through `tags`."""
    c.post('/api/projects', json={'name': name})
    rng = np.random.default_rng(5)
    uploads = []
    for i in range(count):
        img = np.full((352, 416, 3), 38, np.uint8)
        img = cv2.add(img, rng.integers(0, 22, (352, 416, 3), dtype=np.uint8))
        cv2.rectangle(img, (120, 90), (260, 230), (90, 96, 108), -1)
        cv2.putText(img, tags[i % len(tags)][:1], (150, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (240, 248, 255), 8, cv2.LINE_AA)
        uploads.append((io.BytesIO(cv2.imencode('.jpg', img)[1].tobytes()),
                        f'{i:02d}.jpg'))
    r = c.post(f'/api/projects/{name}/images', data={'images': uploads},
               content_type='multipart/form-data')
    names = [x['filename'] for x in r.get_json()['imported']]
    for i, filename in enumerate(names):
        c.post(f'/api/projects/{name}/images/{filename}/annotations',
               json={'regions': [{'tag': tags[i % len(tags)], 'x': 115, 'y': 85,
                                  'width': 150, 'height': 150}]})
    return names


print('== which class lists mean mirroring would be wrong ==')
CASES = [
    (['0', '1', '2', '9'], True, 'single digits'),
    (['A', 'B', 'C'], True, 'single letters'),
    (['10', '25', '100'], True, 'numbers being read off a display'),
    (['cat', 'dog'], False, 'words for objects'),
    (['bolt', 'crack', 'scratch'], False, 'defect names'),
    ([], False, 'no classes yet'),
]
for names, expected, why in CASES:
    got = trainaug.classes_are_orientation_sensitive(names)
    check(f'{why}: {"off" if expected else "on"}', got == expected, f'{names} -> {got}')

print('\n== a digits project ==')
make_project('digits', ['0', '1', '2', '3'])
advice = c.get('/api/projects/digits/training/options').get_json()['augmentation']
check('the advice is offered with the options', bool(advice), advice)
check('it noticed the classes are characters', advice['orientation_sensitive'])
check('mirroring is switched off', advice['settings']['fliplr'] == 0.0,
      advice['settings']['fliplr'])
check('it says why', any('mirror' in r.lower() for r in advice['reasons']),
      advice['reasons'])
check('the other augmentation is left alone',
      advice['settings']['hsv_v'] > 0 and advice['settings']['scale'] > 0,
      advice['settings'])
check('it points out how few images there are',
      any('15' in r for r in advice['reasons']), advice['reasons'])
check('it suggests presets to generate', bool(advice['suggested_presets']),
      advice['suggested_presets'])

print('\n== an objects project ==')
make_project('objects', ['bolt', 'crack'])
other = c.get('/api/projects/objects/training/options').get_json()['augmentation']
check('mirroring stays on where it is harmless', other['settings']['fliplr'] == 0.5,
      other['settings']['fliplr'])
check('and it says so', any('mirror' in r.lower() for r in other['reasons']),
      other['reasons'])

print('\n== the settings reach the run ==')
r = c.post('/api/projects/digits/training/start', json={
    'model_type': 'yolo11n', 'epochs': 1, 'batch_size': 4, 'img_size': 160,
    'model_name': 'aug_check'})
body = r.get_json()
check('the run started', r.status_code == 200, body)
config = body.get('config') or {}
check('the run records what augmentation it will use',
      isinstance(config.get('augmentation'), dict), config.get('augmentation'))
check('and that mirroring is off for it',
      (config.get('augmentation') or {}).get('fliplr') == 0.0,
      config.get('augmentation'))
check('the reasoning is kept with the run', bool(config.get('augmentation_reasons')),
      config.get('augmentation_reasons'))
c.post('/api/projects/digits/training/stop')

print('\n== an override is honoured, and clamped ==')
c.post('/api/projects/digits/training/reset')
r = c.post('/api/projects/digits/training/start', json={
    'model_type': 'yolo11n', 'epochs': 1, 'batch_size': 4, 'img_size': 160,
    'model_name': 'aug_override',
    'augmentation': {'fliplr': 1.0, 'hsv_v': 9.9, 'mosaic': -3, 'nonsense': 'x'}})
config = (r.get_json() or {}).get('config') or {}
settings = config.get('augmentation') or {}
check('an explicit choice overrides the recommendation', settings.get('fliplr') == 1.0,
      settings.get('fliplr'))
check('an out-of-range value is clamped rather than refused',
      settings.get('hsv_v') == 1.0 and settings.get('mosaic') == 0.0,
      (settings.get('hsv_v'), settings.get('mosaic')))
check('an unknown key is dropped', 'nonsense' not in settings, list(settings))
c.post('/api/projects/digits/training/stop')

print('\n== filtered copies generated as part of starting a run ==')
c.post('/api/projects/digits/training/reset')
before = c.get('/api/projects/digits/dataset-summary').get_json()['total_images']
r = c.post('/api/projects/digits/training/start', json={
    'model_type': 'yolo11n', 'epochs': 1, 'batch_size': 4, 'img_size': 160,
    'model_name': 'aug_filters',
    'generate_filters': ['clahe', 'tophat']})
body = r.get_json()
check('the run started with filters requested', r.status_code == 200, body)
config = body.get('config') or {}
after = c.get('/api/projects/digits/dataset-summary').get_json()['total_images']
check('new images were written', after > before, (before, after))
check('two presets over fifteen images is thirty copies',
      config.get('generated_filters') == 30, config.get('generated_filters'))
check('the run records which presets it used',
      set(config.get('generated_filter_presets') or []) == {'clahe', 'tophat'},
      config.get('generated_filter_presets'))
# Not every generated image reaches the dataset, and that is deliberate: a
# copy of an image that landed in validation belongs in neither split. What
# matters is that the count adds up and the shortfall is explained rather than
# leaving someone to wonder where six images went.
held = c.post('/api/projects/digits/prepare-dataset').get_json()['dataset']
accounted = held['train_images'] + held['val_images'] + held['held_back_count']
check('every image on disk is either in a split or explained',
      accounted + held['skipped_count'] == after,
      (held['train_images'], held['val_images'], held['held_back_count'],
       held['skipped_count'], after))
check('the copies of validation images are the ones held back',
      held['held_back_count'] > 0 and
      all('_aug_' in n for n in held['held_back']), held['held_back'][:2])
check('and the report says why', 'validation' in held['held_back_reason'],
      held['held_back_reason'])

print('\n== the copies do not leak across the train/val line ==')
# Every filtered copy of one photograph has to stay on the same side as the
# original, or a variant of a training image ends up scoring the model.
dataset_dir = Path(config['dataset_path'])
train = {p.stem.split('_aug_')[0] for p in (dataset_dir / 'images' / 'train').iterdir()}
val = {p.stem.split('_aug_')[0] for p in (dataset_dir / 'images' / 'val').iterdir()}
check('no source image appears on both sides', not (train & val),
      sorted(train & val)[:3])
c.post('/api/projects/digits/training/stop')

print('\n== how the images reach the GPU ==')
# Both were fixed: workers at 0 and cache off. On a slow card neither matters
# because the GPU is what is being waited on; on a fast one a single decoding
# process becomes the bottleneck and the card idles between batches, which
# looks exactly like a slow GPU and gives no hint of the real cause.
workers, cache = trainaug.loader_settings(2232, 640)
check('loader processes are used', workers > 0, workers)

# Caching the decoded images is deliberately off. Measured twice on the same
# data with the same seed, both runs reported mAP50 0.995 and the cached one
# then scored 0.01 on every held-out image while the uncached one scored
# 0.09-0.33: a model that reports perfectly and detects nothing, which is the
# exact failure this application already shipped once.
check('images are not cached in memory by default', cache is False, cache)
check('not even for a small set',
      trainaug.loader_settings(15, 640)[1] is False)

check('an explicit choice is honoured',
      trainaug.loader_settings(100, 640, workers=0, cache=False) == (0, False))
check('caching remains reachable for anyone re-measuring it',
      trainaug.loader_settings(100, 640, cache='ram')[1] == 'ram')
check('nonsense falls back to the defaults',
      trainaug.loader_settings(100, 640, workers='x', cache='nope')
      == (trainaug.DEFAULT_WORKERS, False))
check('workers cannot be set absurdly high',
      trainaug.loader_settings(100, 640, workers=999)[0] <= 16,
      trainaug.loader_settings(100, 640, workers=999)[0])

advice = c.get('/api/projects/digits/training/options').get_json()['loader']
check('the options carry the loader advice', bool(advice), advice)
check('and say what to do if it deadlocks',
      'epoch 1' in advice.get('note', ''), advice.get('note', '')[:80])

c.post('/api/projects/digits/training/reset')
r = c.post('/api/projects/digits/training/start', json={
    'model_type': 'yolo11n', 'epochs': 1, 'batch_size': 4, 'img_size': 160,
    'model_name': 'loader_check', 'workers': 2})
config = (r.get_json() or {}).get('config') or {}
check('the run records the loader settings it was given',
      config.get('workers') == 2 and config.get('cache') is False,
      (config.get('workers'), config.get('cache')))
c.post('/api/projects/digits/training/stop')

print('\n== presets that only repeat what training already does are not offered ==')
# Brightness and colour are applied fresh every epoch at no cost in disk or
# epoch time, so writing them out as files adds nothing.
overlapping = {'bright', 'dark', 'warm', 'cool', 'vivid', 'sepia', 'gamma_low',
               'gamma_high', 'high_contrast', 'invert', 'gray', 'equalize'}
offered = set(trainaug.STRUCTURAL_PRESETS)
check('the on-the-fly ones are excluded', not (offered & overlapping),
      sorted(offered & overlapping))
check('the structural ones are included',
      {'adaptive_thresh', 'tophat', 'canny_overlay', 'clahe'} <= offered,
      sorted(offered))

print('\n== an unknown preset name is ignored, not fatal ==')
c.post('/api/projects/objects/training/reset')
r = c.post('/api/projects/objects/training/start', json={
    'model_type': 'yolo11n', 'epochs': 1, 'batch_size': 4, 'img_size': 160,
    'model_name': 'aug_bad', 'generate_filters': ['not_a_preset']})
check('a run with only unknown presets still starts', r.status_code == 200,
      r.get_json())
check('and generates nothing',
      ((r.get_json() or {}).get('config') or {}).get('generated_filters') == 0,
      (r.get_json() or {}).get('config', {}).get('generated_filters'))
c.post('/api/projects/objects/training/stop')

print('\n' + ('TRAINING AUGMENTATION OK' if not fails
              else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
