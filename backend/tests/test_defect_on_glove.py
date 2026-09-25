"""
A defect lands on the glove, at the size and the spot it had on its own.

The first attempt put patches of machinery over machinery, nowhere near a
glove. Three causes, all the same mistake wearing different clothes -- the
frame was treated as the thing, when the glove is the thing:

  * the harvested patch was the whole annotation rectangle, so a clamp or a
    piece of the former inside it came across as part of the defect
  * its size was carried in pixels, though the same glove fills a different
    number of them in another camera's framing
  * the glove was looked for by thresholding the whole frame, which works for
    one glove on a plain field and not for a glove among machinery

So the pictures here look like the ones it failed on: a glove occupying a
corner of a frame full of metal. Everything is measured against the box of the
glove, and what is checked is that a defect recorded as "a third of the way
across, a fifth of the glove wide" comes back as exactly that on a glove of a
different size, in a different place, on a different colour of rubber.

    python backend/tests/test_defect_on_glove.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_onglove_'))
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


def scene(glove_box, glove_level, seed, defect=None):
    """
    A glove on a machine: clamps, a former, structure everywhere.

    Deliberately not a glove alone on a plain field. Thresholding this finds
    the machinery as readily as the rubber, which is the situation the first
    attempt could not handle.
    """
    rng = np.random.default_rng(seed)
    img = np.full((360, 560), 30.0, np.float32)

    # Machinery: bright metal, high contrast, all over the frame.
    for x in (40, 180, 330, 480):
        cv2.rectangle(img, (x, 20), (x + 26, 340), 150.0, -1)
    cv2.rectangle(img, (0, 300), (560, 360), 110.0, -1)
    cv2.circle(img, (270, 70), 46, 165.0, -1)

    x, y, w, h = glove_box
    cv2.ellipse(img, (int(x + w / 2), int(y + h / 2)),
                (int(w / 2), int(h / 2)), 0, 0, 360, float(glove_level), -1)

    if defect is not None:
        (cx, cy), radius, strength = defect
        patch = np.zeros_like(img)
        cv2.circle(patch, (int(cx), int(cy)), int(radius), 1.0, -1)
        patch = cv2.GaussianBlur(patch, (9, 9), 2.0)
        img = img * np.exp(-patch * strength)

    img += rng.standard_normal(img.shape).astype(np.float32) * 1.4
    return np.clip(img, 0, 255).astype(np.uint8)


def store(project, name, gray, regions):
    """Write a picture and its boxes straight in, as an import would."""
    path = projects.images_dir(project) / name
    path.write_bytes(cv2.imencode('.png', gray)[1].tobytes())
    projects.write_annotation(project, name, {
        'filename': name, 'regions': regions, 'annotated': bool(regions),
        'width': int(gray.shape[1]), 'height': int(gray.shape[0]),
    })


# ── the source line: pale gloves, every picture boxed ───────────────────────
SOURCE_GLOVE = (120, 90, 180, 200)          # x, y, w, h
PALE = 185

# A defect a third of the way across the glove and a fifth of it wide. Those
# two fractions are the whole of what has to survive the move.
REL_CX, REL_CY, REL_W = 0.33, 0.45, 0.20
SPOT = (SOURCE_GLOVE[0] + REL_CX * SOURCE_GLOVE[2],
        SOURCE_GLOVE[1] + REL_CY * SOURCE_GLOVE[3])
RADIUS = REL_W * SOURCE_GLOVE[2] / 2

print('== a source line, gloves and defects both boxed ==')
with app.app_context():
    projects.create_project('pale-line')
    for i in range(6):
        gray = scene(SOURCE_GLOVE, PALE, 10 + i, (SPOT, RADIUS, 0.6))
        pad = RADIUS + 3
        store('pale-line', f's{i}.png', gray, [
            {'tag': 'Good', 'x': SOURCE_GLOVE[0], 'y': SOURCE_GLOVE[1],
             'width': SOURCE_GLOVE[2], 'height': SOURCE_GLOVE[3]},
            {'tag': 'DirtM', 'x': SPOT[0] - pad, 'y': SPOT[1] - pad,
             'width': pad * 2, 'height': pad * 2},
        ])
    projects.rebuild_index('pale-line')

r = c.post('/api/projects/pale-line/defect-library',
           json={'labels': ['DirtM'], 'per_label': 20})
body = r.get_json() or {}
check('the collection runs', r.status_code == 200, body)
check('it found the defects', body.get('total') == 6, body.get('total'))

meta = defectlib.load('pale-line') or {}
sample = (meta.get('defects') or [{}])[0]
rel = sample.get('rel') or {}
check('each is recorded as a fraction of its glove', bool(rel), sample)
check('the position it sat at',
      abs(rel.get('cx', 0) - REL_CX) < 0.05 and abs(rel.get('cy', 0) - REL_CY) < 0.05,
      rel)
check('and the share of the glove it covered',
      abs(rel.get('w', 0) - REL_W) < 0.06, rel)

print('\n== what was kept is the defect, not the rectangle around it ==')
with np.load(defectlib.library_dir('pale-line') / sample['file']) as data:
    kept_mask = data['mask']
# The annotation was a square; a round blot inside it should not come out
# square, or everything else in that square came too.
filled = float((kept_mask > 128).mean())
check('it was trimmed to the shape, not left as a box',
      filled < 0.85, round(filled, 3))
check('and its edge fades rather than being cut',
      0 < float((kept_mask > 10).mean()) - filled, round(filled, 3))

print('\n== the glove is found among the machinery ==')
# Half the size, elsewhere in the frame, and dark instead of pale.
TARGET_GLOVE = (330, 150, 90, 100)
DARK = 72

with app.app_context():
    projects.create_project('dark-line')
    for i in range(6):
        store('dark-line', f'g{i}.png', scene(TARGET_GLOVE, DARK, 100 + i), [
            {'tag': 'Good', 'x': TARGET_GLOVE[0], 'y': TARGET_GLOVE[1],
             'width': TARGET_GLOVE[2], 'height': TARGET_GLOVE[3]},
        ])
    projects.rebuild_index('dark-line')

    gray = cv2.imdecode(np.frombuffer(
        (projects.images_dir('dark-line') / 'g0.png').read_bytes(), np.uint8),
        cv2.IMREAD_GRAYSCALE)
    box, how = defectlib.find_glove('dark-line', 'g0.png', gray, None, None)
check('its box comes from the label that is already there', how == 'label', how)
check('and it is the glove', box == TARGET_GLOVE, box)

with app.app_context():
    rubber = defectlib.rubber_inside(gray.astype(np.float32), box)
inside = (int(TARGET_GLOVE[1] + TARGET_GLOVE[3] / 2),
          int(TARGET_GLOVE[0] + TARGET_GLOVE[2] / 2))
check('the rubber inside that box is found', rubber[inside] > 0, rubber[inside])
check('and the clamp across the frame is not', rubber[200, 50] == 0,
      rubber[200, 50])
check('nor anything outside the box at all',
      float((rubber[:, :TARGET_GLOVE[0]] > 0).mean()) == 0.0)

print('\n== the defect goes where it belongs, at the size it should be ==')
# The good gloves are already boxed, so nothing here needs a model. That is
# the point of importing a dataset with Good on every picture.
with app.app_context():
    # Left exactly as an import leaves them: boxed as Good, and therefore
    # counted as annotated. A good glove in an export is not an unlabelled
    # picture -- it carries the box that says where the glove is, which is the
    # box this needs. Treating any boxed picture as already dealt with is what
    # left nothing to work on.
    pass

r = c.post('/api/projects/dark-line/defect-synth',
           json={'source_project': 'pale-line', 'tags': ['DirtM'],
                 'per_image': 1})
check('the run starts', r.status_code == 200, r.get_json())

with app.app_context():
    status = defectlib.wait_for_idle('dark-line', timeout=240)
print(f"    {status.get('message')}")
check('it finishes', status.get('status') == 'finished', status)
check('and made something', (status.get('made') or 0) >= 3, status.get('made'))

# The good gloves are annotated too -- they carry their Good box -- so the
# made ones are told apart by being marked as made, not by having boxes.
with app.app_context():
    made = [projects.read_annotation('dark-line', e['filename'])
            for e in projects.list_images('dark-line')
            if e.get('synthetic')]

check('every made picture is boxed', len(made) == status.get('made'),
      (len(made), status.get('made')))
check('and the good gloves are still there, untouched',
      len([e for e in projects.list_images('dark-line')
           if not e.get('synthetic')]) == 6,
      len([e for e in projects.list_images('dark-line') if not e.get('synthetic')]))

if made:
    box_on = (made[0].get('regions') or [{}])[0]
    cx = box_on['x'] + box_on['width'] / 2
    cy = box_on['y'] + box_on['height'] / 2
    gx, gy, gw, gh = TARGET_GLOVE
    print(f"    landed at ({cx:.0f}, {cy:.0f}); the glove is "
          f"{gx}..{gx + gw} across, {gy}..{gy + gh} down")

    check('it landed on the glove, not on the machinery',
          gx <= cx <= gx + gw and gy <= cy <= gy + gh, (cx, cy))
    check('at the same place on it as before',
          abs((cx - gx) / gw - REL_CX) < 0.18
          and abs((cy - gy) / gh - REL_CY) < 0.18,
          ((cx - gx) / gw, (cy - gy) / gh))
    check('and at the same share of it',
          abs(box_on['width'] / gw - REL_W) < 0.15,
          box_on['width'] / gw)
    check('which is fewer pixels than on the larger glove it came from',
          box_on['width'] < REL_W * SOURCE_GLOVE[2],
          (box_on['width'], REL_W * SOURCE_GLOVE[2]))

print('\n== a picture with no glove in it is skipped, not guessed at ==')
with app.app_context():
    projects.create_project('no-gloves')
    for i in range(3):
        store('no-gloves', f'm{i}.png', scene((0, 0, 1, 1), 30, 400 + i), [])
    projects.rebuild_index('no-gloves')

r = c.post('/api/projects/no-gloves/defect-synth',
           json={'source_project': 'pale-line', 'tags': ['DirtM']})
with app.app_context():
    status = defectlib.wait_for_idle('no-gloves', timeout=180)
check('nothing is made', (status.get('made') or 0) == 0, status.get('made'))
check('and it says why rather than failing silently',
      (status.get('no_glove') or 0) == 3, status)
check('with the count in the message',
      'no glove' in (status.get('message') or ''), status.get('message'))

print('\n== one export with both kinds in it ==')
# A dataset does not come sorted. The same camera photographs good gloves and
# bad ones all shift, and the export holds them together: a picture with only
# a Good box on it is a good glove, and one with a defect box as well is not.
# The system has to tell them apart itself, because nobody is going to split
# six thousand files by hand.
with app.app_context():
    projects.create_project('mixed-line')
    for i in range(5):
        store('mixed-line', f'ok{i}.png', scene(TARGET_GLOVE, DARK, 500 + i), [
            {'tag': 'Good', 'x': TARGET_GLOVE[0], 'y': TARGET_GLOVE[1],
             'width': TARGET_GLOVE[2], 'height': TARGET_GLOVE[3]},
        ])
    for i in range(3):
        spot = (TARGET_GLOVE[0] + 30, TARGET_GLOVE[1] + 40)
        gray = scene(TARGET_GLOVE, DARK, 600 + i, (spot, 9, 0.6))
        store('mixed-line', f'bad{i}.png', gray, [
            {'tag': 'Good', 'x': TARGET_GLOVE[0], 'y': TARGET_GLOVE[1],
             'width': TARGET_GLOVE[2], 'height': TARGET_GLOVE[3]},
            {'tag': 'HoleLeft', 'x': spot[0] - 12, 'y': spot[1] - 12,
             'width': 24, 'height': 24},
        ])
    projects.rebuild_index('mixed-line')

r = c.post('/api/projects/mixed-line/defect-synth',
           json={'source_project': 'pale-line', 'tags': ['DirtM'],
                 'per_image': 1})
check('the run starts on a mixed project', r.status_code == 200, r.get_json())

with app.app_context():
    status = defectlib.wait_for_idle('mixed-line', timeout=240)
print(f"    {status.get('message')}")
check('it works on the five good gloves',
      (status.get('made') or 0) == 5, status.get('made'))

with app.app_context():
    after = projects.list_images('mixed-line')
check('the three already defective are left alone',
      len([e for e in after if not e.get('synthetic')]) == 8, len(after))
check('and the ones it made are marked as made',
      len([e for e in after if e.get('synthetic')]) == 5,
      len([e for e in after if e.get('synthetic')]))

print('\n== a project where every picture is already defective ==')
with app.app_context():
    projects.create_project('all-bad')
    for i in range(3):
        spot = (TARGET_GLOVE[0] + 30, TARGET_GLOVE[1] + 40)
        gray = scene(TARGET_GLOVE, DARK, 800 + i, (spot, 9, 0.6))
        store('all-bad', f'b{i}.png', gray, [
            {'tag': 'Good', 'x': TARGET_GLOVE[0], 'y': TARGET_GLOVE[1],
             'width': TARGET_GLOVE[2], 'height': TARGET_GLOVE[3]},
            {'tag': 'HoleLeft', 'x': spot[0] - 12, 'y': spot[1] - 12,
             'width': 24, 'height': 24},
        ])
    projects.rebuild_index('all-bad')

r = c.post('/api/projects/all-bad/defect-synth',
           json={'source_project': 'pale-line', 'tags': ['DirtM']})
message = (r.get_json() or {}).get('message', '')
check('says they already have a defect, not that the project is empty',
      r.status_code == 400 and 'already have a defect' in message,
      (r.status_code, message[:120]))

print('\n== a class that describes the whole glove ==')
# NoFormer, NonStrip, OpenTop: the box is the glove, because the statement is
# about the glove. Nothing can be pasted onto a good one to make it true, and
# offering those beside Hole and DirtM invites somebody to tick them and get
# nonsense back -- which is most of what came back.
with app.app_context():
    for i in range(4):
        gray = scene(SOURCE_GLOVE, PALE, 700 + i)
        store('pale-line', f'w{i}.png', gray, [
            {'tag': 'Good', 'x': SOURCE_GLOVE[0], 'y': SOURCE_GLOVE[1],
             'width': SOURCE_GLOVE[2], 'height': SOURCE_GLOVE[3]},
            {'tag': 'NoFormer', 'x': SOURCE_GLOVE[0] + 4,
             'y': SOURCE_GLOVE[1] + 4,
             'width': SOURCE_GLOVE[2] - 8, 'height': SOURCE_GLOVE[3] - 8},
        ])
    projects.rebuild_index('pale-line')

r = c.post('/api/projects/pale-line/defect-library',
           json={'labels': ['DirtM', 'NoFormer'], 'per_label': 20})
rows = {row['tag']: row for row in ((r.get_json() or {}).get('per_tag') or [])}
check('the mark on the rubber is usable',
      rows.get('DirtM', {}).get('usable') is True, rows.get('DirtM'))
check('and the whole-glove one is not',
      rows.get('NoFormer', {}).get('whole_glove') is True, rows.get('NoFormer'))
check('told apart by how much of the glove it covers',
      rows.get('NoFormer', {}).get('share', 0)
      > rows.get('DirtM', {}).get('share', 1),
      (rows.get('DirtM', {}).get('share'), rows.get('NoFormer', {}).get('share')))

r = c.post('/api/projects/dark-line/defect-synth',
           json={'source_project': 'pale-line', 'tags': ['NoFormer']})
message = (r.get_json() or {}).get('message', '')
check('choosing it is refused, with the reason',
      r.status_code == 400 and 'whole glove' in message,
      (r.status_code, message[:130]))

print('\n' + ('DEFECT ON GLOVE OK' if not fails else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
