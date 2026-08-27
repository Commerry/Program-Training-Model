"""
Colour-tone augmentation, checked properly.

The workflow this supports is: annotate a handful of images, let the system
generate many filtered copies so the detector has more to learn from. That only
works if two things hold, and both are worth verifying rather than assuming.

  1. The filters are per-pixel or convolution operations with no geometric
     component, so the boxes drawn on the source stay correct on every copy.
     A preset that resized, cropped or shifted would silently produce thousands
     of mislabelled training images.

  2. Each copy still shows the annotated object. A filter that suits one
     dataset can erase the object in another, and an image whose label points
     at nothing teaches the detector that the class looks like empty background.
"""
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TMP = Path(tempfile.mkdtemp(prefix='vt_aug_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
from app import create_app                              # noqa: E402
from services import augment                            # noqa: E402

app = create_app()
c = app.test_client()

fails = []


def check(label, cond, detail=''):
    print(('  PASS ' if cond else '  FAIL ') + label + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


def mean_value(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()


def edge_energy(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    return cv2.Laplacian(grey, cv2.CV_64F).var()


ROOT = Path(os.environ['PROJECTS_ROOT'])
c.post('/api/projects', json={'name': 'aug'})

# Ten sources, each a bright digit on a darker plate at a recorded position.
# Ten rather than a handful so the batch-size limit further down is reachable:
# 10 x 26 presets x 20 variants exceeds the cap, four sources would not.
SHAPES = [(640, 480), (480, 640), (512, 512)]
rng = np.random.default_rng(3)
uploads, truth = [], []
for i in range(10):
    w, h = SHAPES[i % len(SHAPES)]
    img = np.full((h, w, 3), 40, np.uint8)
    img = cv2.add(img, rng.integers(0, 25, (h, w, 3), dtype=np.uint8))
    x, y, bw, bh = 90 + i * 15, 70, 80, 110
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (15, 18, 26), -1)
    cv2.putText(img, str(i % 10), (x + 12, y + bh - 22), cv2.FONT_HERSHEY_SIMPLEX,
                2.6, (240, 248, 255), 6, cv2.LINE_AA)
    uploads.append((io.BytesIO(cv2.imencode('.jpg', img)[1].tobytes()), f'src{i}.jpg'))
    truth.append((w, h, x, y, bw, bh, str(i % 10)))

r = c.post('/api/projects/aug/images', data={'images': uploads},
           content_type='multipart/form-data')
names = [x['filename'] for x in r.get_json()['imported']]
for name, (w, h, x, y, bw, bh, tag) in zip(names, truth):
    c.post(f'/api/projects/aug/images/{name}/annotations',
           json={'regions': [{'tag': tag, 'x': x, 'y': y, 'width': bw, 'height': bh}]})

print('== starting point ==')
s0 = c.get('/api/projects/aug/dataset-summary').get_json()
check('10 source images, all annotated',
      s0['total_images'] == 10 and s0['annotated_images'] == 10, s0['total_images'])

print('\n== every preset is offered ==')
tones = c.get('/api/projects/aug/augment-color/tones').get_json()['tones']
check('26 presets exposed', len(tones) == 26, len(tones))
check('service and API agree', tones == augment.ALL_COLOR_TONES)

print('\n== generate one variant per preset ==')
r = c.post('/api/projects/aug/augment-color',
           json={'variants_per_tone': 1, 'strength': 1.0})
body = r.get_json()
check('request accepted', r.status_code == 200, body.get('message'))
expected = 10 * 26 * 1
check('reports what it planned', body.get('planned') == expected, body.get('planned'))
check('every planned image is either created or explained',
      body['created_count'] + len(body.get('skipped', [])) == expected,
      (body['created_count'], len(body.get('skipped', []))))
# On data every preset suits, the survival guard should not be dropping any.
check('no preset hid the object on this data',
      not body.get('dropped_by_tone'), body.get('dropped_by_tone'))
check(f'created {expected} images (10 sources x 26 presets)',
      body['created_count'] == expected, body['created_count'])

print('\n== the generated images are usable ==')
images = c.get('/api/projects/aug/images').get_json()['images']
aug = [i for i in images if i['augmented']]
src = [i for i in images if not i['augmented']]
check('sources are untouched', len(src) == 10, len(src))
check('all variants are marked augmented', len(aug) == expected, len(aug))
check('every variant carries the annotation',
      all(i['annotated'] and i['boxes'] for i in aug),
      len([i for i in aug if not i['boxes']]))

print('\n== geometry: the filters must not move anything ==')
by_source = {}
for entry in aug:
    ann = json.loads((ROOT / 'aug' / 'annotations' / f'{entry["filename"]}.json')
                     .read_text(encoding='utf-8'))
    by_source.setdefault(ann['augmentation']['source_image'], []).append((entry, ann))

wrong_size, moved_box = [], []
for source_name, variants in by_source.items():
    index = names.index(source_name)
    w, h, x, y, bw, bh, tag = truth[index]
    for entry, ann in variants:
        if entry['width'] != w or entry['height'] != h:
            wrong_size.append((entry['filename'], entry['width'], entry['height'], w, h))
        box = ann['regions'][0]
        if (round(box['x']), round(box['y']), round(box['width']),
                round(box['height'])) != (x, y, bw, bh):
            moved_box.append((entry['filename'], box))

check('every variant keeps the source resolution', not wrong_size, wrong_size[:2])
check('every box keeps its exact coordinates', not moved_box, moved_box[:2])

print('\n== the object is still inside the box after filtering ==')
# Whatever a preset does to the colours, the region inside the box must still
# differ measurably from the band around it, or the box now labels a patch of
# plain background. The bar is relative to the source: a box round a low
# contrast object starts out barely separated, and demanding a fixed contrast
# would throw away good data.
lost = []
for source_name, variants in by_source.items():
    index = names.index(source_name)
    w, h, x, y, bw, bh, tag = truth[index]
    source_img = cv2.imread(str(ROOT / 'aug' / 'images' / source_name))
    baseline = augment._region_separation(source_img, (x, y, bw, bh))
    for entry, ann in variants:
        img = cv2.imread(str(ROOT / 'aug' / 'images' / entry['filename']))
        if img is None:
            lost.append((entry['filename'], 'unreadable'))
            continue
        after = augment._region_separation(img, (x, y, bw, bh))
        keeps = (after[0] >= baseline[0] * augment.RETAIN_FRACTION
                 or after[1] >= baseline[1] * augment.RETAIN_FRACTION)
        if not keeps:
            lost.append((ann['augmentation']['tone'], entry['filename'],
                         round(after[0], 1), round(after[1], 1)))

check('the annotated region still stands out in every variant', not lost, lost[:4])

print('\n== each preset produces a genuinely different image ==')
# Compared in COLOUR: a grayscale fingerprint cannot tell warm from cool. The
# comparison also looks at edge energy and saturation, because downsampling to
# a thumbnail discards exactly what the detail presets change.
first_source = names[0]


def fingerprint(image):
    return (cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(int),
            edge_energy(image),
            cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].mean())


def differ(a, b):
    """Different along any of the axes the presets are meant to move."""
    return (np.abs(a[0] - b[0]).mean() > 1.5
            or abs(a[1] - b[1]) / max(a[1], b[1], 1e-6) > 0.15
            or abs(a[2] - b[2]) > 3)


prints = {ann['augmentation']['tone']:
          fingerprint(cv2.imread(str(ROOT / 'aug' / 'images' / entry['filename'])))
          for entry, ann in by_source[first_source]}

source_img = cv2.imread(str(ROOT / 'aug' / 'images' / first_source))
source_print = fingerprint(source_img)

no_ops = [t for t, f in prints.items() if not differ(f, source_print)]
check('no preset is a no-op', not no_ops, no_ops)

tone_list = list(prints)
duplicates = [(tone_list[i], tone_list[j])
              for i in range(len(tone_list))
              for j in range(i + 1, len(tone_list))
              if not differ(prints[tone_list[i]], prints[tone_list[j]])]
check('no two presets produce the same image', not duplicates, duplicates[:3])

print('\n== the two gamma presets are not the same filter ==')
# They used to compute the same exponent and emit identical images, wasting a
# preset and putting a duplicate row in the dataset for every source.
low = augment.apply_color_tone(source_img, 'gamma_low', 1.0, 1)
high = augment.apply_color_tone(source_img, 'gamma_high', 1.0, 1)
check('gamma_low and gamma_high differ', not np.array_equal(low, high))
check('gamma_low darkens', mean_value(low) < mean_value(source_img) - 5,
      f'{mean_value(source_img):.1f} -> {mean_value(low):.1f}')
check('gamma_high brightens', mean_value(high) > mean_value(source_img) + 5,
      f'{mean_value(source_img):.1f} -> {mean_value(high):.1f}')

print('\n== sharpen sharpens without changing exposure ==')
# The kernel used to sum to 0.5 rather than 1, so it halved the brightness of
# every image it touched on top of sharpening it.
sharp = augment.apply_color_tone(source_img, 'sharpen', 1.0, 1)
shift = abs(mean_value(sharp) - mean_value(source_img)) / mean_value(source_img)
check('brightness is preserved', shift < 0.08, f'{shift * 100:.1f}% shift')
check('edges are enhanced', edge_energy(sharp) > edge_energy(source_img) * 1.5,
      f'{edge_energy(source_img):.0f} -> {edge_energy(sharp):.0f}')

print('\n== a variant that hides the object is dropped, not written ==')
real_apply = augment.apply_color_tone


def erase_on_one_tone(image, tone, strength=1.0, seed=None):
    # Stands in for a preset that does not suit the data: flat grey, object gone.
    if tone == 'sepia':
        return np.full_like(image, 128)
    return real_apply(image, tone, strength, seed)


augment.apply_color_tone = erase_on_one_tone
try:
    guarded = c.post('/api/projects/aug/augment-color',
                     json={'variants_per_tone': 1,
                           'tones': ['sepia', 'bright']}).get_json()
finally:
    augment.apply_color_tone = real_apply

check('the destructive preset was dropped for every source',
      guarded.get('dropped_by_tone', {}).get('sepia') == 10,
      guarded.get('dropped_by_tone'))
check('the sound preset alongside it was kept',
      guarded['created_count'] == 10, guarded['created_count'])
check('the report says which preset was dropped and why',
      'sepia' in guarded['message'] and 'hid the annotated object' in guarded['message'],
      guarded['message'])

print('\n== augmented copies must not leak into validation ==')
report = c.post('/api/projects/aug/prepare-dataset').get_json()['dataset']
val_dir = Path(report['dataset_path']) / 'images' / 'val'
leaked = [p.name for p in val_dir.iterdir() if '_aug_' in p.name]
check('validation holds no generated image', not leaked, leaked[:3])
print(f"    {report['train_images']} train / {report['val_images']} val")

print('\n== a second run does not augment the augmented ==')
b2 = c.post('/api/projects/aug/augment-color',
            json={'variants_per_tone': 1, 'tones': ['warm']}).get_json()
check('still only the 10 originals are used as sources',
      b2.get('source_count') == 10, b2.get('source_count'))

print('\n== the limit refuses an unreasonable request with the arithmetic ==')
# 10 sources x 26 presets x 20 variants = 5,200, over the 5,000 cap.
r3 = c.post('/api/projects/aug/augment-color', json={'variants_per_tone': 20})
check('over-large request is refused', r3.status_code == 400, r3.status_code)
msg = (r3.get_json() or {}).get('message', '')
check('and explains the numbers', 'would generate' in msg and '5,200' in msg, msg[:120])

print('\n' + ('AUGMENTATION OK' if not fails else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {ROOT / "aug"})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
