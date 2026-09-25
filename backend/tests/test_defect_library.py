"""
Collecting defects from one glove colour and putting them on another.

The whole point is that a mono camera turns rubber colour into one grey level,
so a model trained on white gloves is useless on black ones and every colour
on every line needs its own dataset. The defects are the rare half of that
dataset, and waiting for enough torn black gloves is what makes a new line
slow to start.

What is checked here is the round trip. A source project holds labelled
defects on light gloves. A target project holds nothing but good dark gloves.
After collecting and synthesising, the target has defect images with boxes
already on them, ready to train, and the greys in those images are the dark
glove's own.

    python backend/tests/test_defect_library.py
"""
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_defect_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402

import config                                           # noqa: E402
from app import create_app                              # noqa: E402
from services import defectlib, projects                # noqa: E402

config.REPO_ROOT = TMP
defectlib.DEFECT_LIBRARY_DIR = TMP / 'defect-library'

app = create_app()
c = app.test_client()

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


BG = 232          # backlight
LIGHT = 175       # a pale glove, where the defects were collected
DARK = 72         # a dark glove on the new line


def frame(level, seed, defect=None):
    """A backlit glove of one grey, optionally with a real defect in it."""
    rng = np.random.default_rng(seed)
    img = np.full((360, 360), float(BG), np.float32)
    cv2.ellipse(img, (180, 190), (110, 140), 0, 0, 360, float(level), -1)
    if defect is not None:
        kind, (cx, cy), radius, strength = defect
        patch = np.zeros_like(img)
        cv2.circle(patch, (cx, cy), radius, 1.0, -1)
        patch = cv2.GaussianBlur(patch, (9, 9), 2.0)
        if kind == 'hole':
            img = img * (1 - patch) + BG * patch
        else:
            # A stain is extra optical density, which is a multiply in
            # intensity -- the same physics the synthesiser has to reproduce.
            img = img * np.exp(-patch * strength)
    img += rng.standard_normal(img.shape).astype(np.float32) * 1.4
    return np.clip(img, 0, 255).astype(np.uint8)


def upload(project, images):
    files = []
    for name, gray in images:
        files.append((io.BytesIO(cv2.imencode('.png', gray)[1].tobytes()), name))
    return c.post(f'/api/projects/{project}/images',
                  data={'images': files}, content_type='multipart/form-data')


print('== a source project of pale gloves with real defects ==')
c.post('/api/projects', json={'name': 'pale-line'})

sources = []
for i in range(6):
    spot = (150 + i * 8, 170 + i * 6)
    sources.append((f'stain{i}.png',
                    frame(LIGHT, 10 + i, ('stain', spot, 16, 0.55)), spot, 16,
                    'DirtM'))
for i in range(4):
    spot = (200 - i * 7, 210 + i * 5)
    sources.append((f'hole{i}.png',
                    frame(LIGHT, 40 + i, ('hole', spot, 11, 0)), spot, 11,
                    'Hole'))

# The upload reply is what carries original_name; the gallery listing does
# not, since nothing in the interface shows it.
reply = upload('pale-line', [(n, g) for n, g, *_ in sources]).get_json() or {}
by_original = {i['original_name']: i['filename'] for i in reply.get('imported') or []}
listed = (c.get('/api/projects/pale-line/images').get_json() or {}).get('images') or []
check('the source images are in', len(listed) == 10, len(listed))

# The glove is boxed as well as the defect. Everything a defect carries with
# it is a fraction of this box -- where on the glove it sat and how much of it
# it covered -- and those two fractions are what survive the move to a glove
# of another size.
GLOVE_BOX = {'tag': 'Good', 'x': 70, 'y': 50, 'width': 220, 'height': 280}

with app.app_context():
    for name, _gray, (cx, cy), radius, tag in sources:
        stored_name = by_original[name]
        pad = radius + 4
        c.post(f'/api/projects/pale-line/images/{stored_name}/annotations',
               json={'regions': [
                   dict(GLOVE_BOX),
                   {'tag': tag, 'x': cx - pad, 'y': cy - pad,
                    'width': pad * 2, 'height': pad * 2},
               ]})

print('\n== collecting their optical signatures ==')
r = c.post('/api/projects/pale-line/defect-library',
           json={'labels': ['DirtM', 'Hole'], 'per_label': 20,
                 'settings': {'src_psf_sigma': 1.0}})
body = r.get_json() or {}
check('the collection runs', r.status_code == 200, body)
check('it found every box', body.get('total') == 10, body.get('total'))

per_tag = {row['tag']: row for row in (body.get('per_tag') or [])}
check('both labels are in the library',
      set(per_tag) == {'DirtM', 'Hole'}, sorted(per_tag))
check('a hole is recognised as light coming through',
      per_tag.get('Hole', {}).get('defect_class') == 'hole', per_tag.get('Hole'))
check('and dirt as something lying on the rubber',
      per_tag.get('DirtM', {}).get('defect_class') == 'stain', per_tag.get('DirtM'))

print('\n== a new line, dark gloves, no defects at all ==')
c.post('/api/projects', json={'name': 'dark-line'})
good_reply = upload('dark-line',
                    [(f'good{i}.png', frame(DARK, 100 + i)) for i in range(8)])
# A good glove is not an annotated picture, but the run still has to know
# where the glove is. The box is written without marking the picture as
# labelled, which is what a dataset with Good on every frame gives you.
with app.app_context():
    for item in (good_reply.get_json() or {}).get('imported') or []:
        record = projects.read_annotation('dark-line', item['filename']) or {}
        record['regions'] = [dict(GLOVE_BOX)]
        record['annotated'] = False
        projects.write_annotation('dark-line', item['filename'], record)
    projects.rebuild_index('dark-line')

before = (c.get('/api/projects/dark-line/images').get_json() or {}).get('images') or []
check('eight good gloves, none of them labelled',
      len(before) == 8 and not any(i.get('annotated') for i in before),
      [(i['filename'], i.get('annotated')) for i in before[:2]])

print('\n== measuring the new line from its own pictures ==')
with app.app_context():
    grays = [cv2.imdecode(np.frombuffer(
        (projects.images_dir('dark-line') / i['filename']).read_bytes(), np.uint8),
        cv2.IMREAD_GRAYSCALE).astype(np.float32) for i in before]
    profile = defectlib.estimate_profile(grays, {})
print(f"    glove {profile['glove_level']:.0f}  backlight {profile['bg_level']:.0f}  "
      f"noise {profile['noise_sigma']:.2f}  k_shot {profile['k_shot']:.4f}")
check('it read the dark glove, not the backlight',
      abs(profile['glove_level'] - DARK) < 12, profile['glove_level'])
check('and the backlight behind it',
      abs(profile['bg_level'] - BG) < 12, profile['bg_level'])
check('noise is measured, not assumed',
      0.5 < profile['noise_sigma'] < 5.0, profile['noise_sigma'])

print('\n== putting the pale line\'s defects on the dark line ==')
r = c.post('/api/projects/dark-line/defect-synth',
           json={'source_project': 'pale-line', 'tags': ['DirtM', 'Hole'],
                 'per_image': 1})
check('the job starts', r.status_code == 200, r.get_json())

with app.app_context():
    status = defectlib.wait_for_idle('dark-line', timeout=180)
print(f"    {status.get('message')}")
check('it finishes', status.get('status') == 'finished', status)
check('it made images', (status.get('made') or 0) >= 5, status.get('made'))

after = (c.get('/api/projects/dark-line/images').get_json() or {}).get('images') or []
made = [i for i in after if i.get('annotated')]
check('every made image came out boxed', len(made) == status.get('made'),
      (len(made), status.get('made')))
check('and the good gloves are still there, still unlabelled',
      len([i for i in after if not i.get('annotated')]) == 8,
      len([i for i in after if not i.get('annotated')]))

print('\n== the boxes are usable for training as they are ==')
with app.app_context():
    records = [projects.read_annotation('dark-line', i['filename']) for i in made]

boxed = [r for r in records if (r.get('regions') or [])]
check('each has at least one box', len(boxed) == len(records), len(boxed))

sample = boxed[0]
region = sample['regions'][0]
check('the box is inside the picture',
      0 <= region['x'] and 0 <= region['y']
      and region['x'] + region['width'] <= sample['width']
      and region['y'] + region['height'] <= sample['height'], region)
check('it is named after the defect, not a number',
      region['tag'] in {'DirtM', 'Hole'}, region['tag'])
check('and it carries the signal strength it was accepted at',
      isinstance(region.get('snr'), float) and region['snr'] >= 3.0,
      region.get('snr'))

print('\n== it is marked as made, not seen ==')
check('the record says so', bool(sample.get('synthetic')), list(sample))
check('naming where the defect came from',
      (sample.get('synthetic') or {}).get('source_project') == 'pale-line',
      sample.get('synthetic'))
check('it is not marked auto-labelled', not sample.get('auto_labelled'))
queue = c.get('/api/projects/dark-line/review/queue').get_json() or {}
check('so nothing goes to be reviewed -- the boxes are exact',
      queue.get('waiting') == 0, queue.get('waiting'))

print('\n== the greys are the dark glove\'s own ==')
with app.app_context():
    made_img = cv2.imdecode(np.frombuffer(
        (projects.images_dir('dark-line') / sample['filename']).read_bytes(),
        np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    source_img = cv2.imdecode(np.frombuffer(
        (projects.images_dir('dark-line') / sample['original_name']).read_bytes(),
        np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Away from the defect the picture has to be the good glove it started as.
corner = np.s_[0:60, 0:60]
check('outside the defect it is still that photograph',
      float(np.abs(made_img[corner] - source_img[corner]).max()) <= 2,
      float(np.abs(made_img[corner] - source_img[corner]).max()))

# And the rubber around the defect reads as the dark glove, not the pale one
# the defect was collected from.
ring = made_img[(made_img > 20) & (made_img < 150)]
check('the rubber still reads dark, not pale',
      abs(float(np.median(ring)) - DARK) < 25, round(float(np.median(ring)), 1))

print('\n== a glove lit from the front, not from behind ==')
# The other ordinary way to photograph one: a bright glove on a dark stage,
# rather than a dark silhouette against a backlight. The code assumed the
# first and took the darker side as the glove every time, so on this kind it
# measured the background as the glove (16, nearly black), the glove as the
# backlight (112), and noise as zero -- which also disabled the gate, since
# nothing can be too faint against no noise. Then it put every defect on the
# background, where it teaches a detector to look at the machine.
def lit_frame(level, seed, defect=None):
    """A bright glove on a dark stage, which is the inverse of the above."""
    rng = np.random.default_rng(seed)
    img = np.full((360, 360), 22.0, np.float32)
    cv2.ellipse(img, (180, 190), (110, 140), 0, 0, 360, float(level), -1)
    if defect is not None:
        kind, (cx, cy), radius, strength = defect
        patch = np.zeros_like(img)
        cv2.circle(patch, (cx, cy), radius, 1.0, -1)
        patch = cv2.GaussianBlur(patch, (9, 9), 2.0)
        img = img * np.exp(-patch * strength)
    img += rng.standard_normal(img.shape).astype(np.float32) * 1.4
    return np.clip(img, 0, 255).astype(np.uint8)


LIT = 150
lit = [lit_frame(LIT, 300 + i) for i in range(8)]
with app.app_context():
    lit_profile = defectlib.estimate_profile([g.astype(np.float32) for g in lit], {})
print(f"    glove {lit_profile['glove_level']:.0f}  "
      f"background {lit_profile['bg_level']:.0f}  "
      f"noise {lit_profile['noise_sigma']:.2f}  k_shot {lit_profile['k_shot']:.4f}")
check('the bright glove is read as the glove',
      abs(lit_profile['glove_level'] - LIT) < 15, lit_profile['glove_level'])
check('and the dark stage as the background',
      lit_profile['bg_level'] < 60, lit_profile['bg_level'])
check('noise is measured on the rubber, not on the empty stage',
      lit_profile['noise_sigma'] > 0.5, lit_profile['noise_sigma'])
check('and never reported as nothing, which would disable the gate',
      lit_profile['k_shot'] > 0, lit_profile['k_shot'])

with app.app_context():
    area = defectlib.glove_area(lit[0].astype(np.float32))
covered = float((area > 0).mean())
check('the area a defect may land on is the glove',
      0.15 < covered < 0.5, round(covered, 3))
# The ellipse sits at (180, 190) with radii 110x140, so its middle is inside
# and a corner of the frame is not.
check('its middle is inside that area', area[190, 180] > 0)
check('and the corner of the frame is not', area[5, 5] == 0)

print('\n== and the backlit kind still reads the other way round ==')
with app.app_context():
    back_area = defectlib.glove_area(frame(DARK, 7).astype(np.float32))
check('the dark glove is still the glove',
      back_area[190, 180] > 0 and back_area[5, 5] == 0,
      (int(back_area[190, 180]), int(back_area[5, 5])))

print('\n== a library that has not been collected ==')
c.post('/api/projects', json={'name': 'empty-line'})
r = c.post('/api/projects/dark-line/defect-synth',
           json={'source_project': 'empty-line', 'tags': []})
check('is refused with what to do about it',
      r.status_code == 400 and 'library' in (r.get_json() or {}).get('message', ''),
      (r.status_code, (r.get_json() or {}).get('message', '')[:100]))

r = c.post('/api/projects/dark-line/defect-synth', json={'tags': []})
check('and so is no source project at all', r.status_code == 400, r.status_code)

print('\n== nothing good left to work on ==')
r = c.post('/api/projects/dark-line/defect-synth',
           json={'source_project': 'pale-line', 'tags': ['DirtM']})
with app.app_context():
    status = defectlib.wait_for_idle('dark-line', timeout=180)
# Every good glove now has a synthetic twin, but the originals are still
# unlabelled, so a second run is legitimate and should simply work again.
check('a second run is allowed and does something',
      status.get('status') == 'finished' and (status.get('made') or 0) > 0,
      status.get('message'))
check('and it is a new batch', (status.get('batch') or 0) >= 2, status.get('batch'))

print('\n' + ('DEFECT LIBRARY OK' if not fails else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
