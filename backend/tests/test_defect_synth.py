"""
Moving a defect between glove colours without moving any pixels.

A mono camera sees a glove as one grey level set by the rubber's colour, so a
model trained on white gloves is useless on black ones and every colour on
every line needs its own dataset. Defects are the rare half of that, and
waiting for enough torn black gloves is what makes a new line slow to start.

A tear is a tear whatever the rubber behind it; what differs is how much light
that rubber passes. So the defect can be moved, provided it is moved in the
domain where it is additive -- optical density, since I = I0 exp(-OD). Copying
pixels would carry the source glove's own brightness across and land the wrong
grey.

What is checked here is exactly that claim, and the consequences that follow
from it:

  * no pixel is imported: everything outside the defect is untouched, and
    inside it the values are the target's own, adjusted
  * the same defect on four different glove greys produces the same change in
    OD, not the same change in grey
  * a hole is the exception, because it is not thinner rubber but no rubber:
    the camera sees the backlight, which is one brightness for every colour
  * the box comes from what is visible after the insert, so it shrinks on a
    dark glove where the same defect shows less
  * a defect too faint to see is refused rather than written into a dataset

    python backend/tests/test_defect_synth.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402

from services import defectsynth                        # noqa: E402

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


BG_LEVEL = 230.0        # backlight seen where there is no glove
SIZE = 400


def glove(level, noise=1.5, seed=1):
    """A flat glove of one grey, on a bright backlight, with sensor noise."""
    rng = np.random.default_rng(seed)
    frame = np.full((SIZE, SIZE), BG_LEVEL, np.float32)
    frame[60:340, 60:340] = level
    frame += rng.standard_normal(frame.shape).astype(np.float32) * noise
    return np.clip(frame, 0, 255).astype(np.uint8)


def profile_for(level, **extra):
    settings = {
        'glove_level': float(level),
        'bg_level': BG_LEVEL,
        'k_shot': 0.02,
        'noise_sigma': 1.5,
        'psf_sigma': 1.0,
        'mm_per_px': 0.05,
        'texture_bank': [],
    }
    settings.update(extra)
    return settings


def stain(strength=0.45, radius=18):
    """A patch of extra optical density -- something lying on the rubber."""
    size = radius * 2 + 9
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    centre = size / 2.0
    dist = np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)
    soft = np.clip(1.0 - dist / radius, 0.0, 1.0)
    return {
        'delta_od': (soft * strength).astype(np.float32),
        'mask': (soft > 0.05).astype(np.uint8) * 255,
        'label_id': 3,
        'defect_class': 'stain',
        'src_mm_per_px': 0.05,
        'src_psf_sigma': 1.0,
    }


def hole(radius=14):
    size = radius * 2 + 9
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    centre = size / 2.0
    dist = np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)
    return {
        'delta_od': np.zeros((size, size), np.float32),
        'mask': ((dist < radius).astype(np.uint8) * 255),
        'label_id': 9,
        'defect_class': 'hole',
        'src_mm_per_px': 0.05,
        'src_psf_sigma': 1.0,
    }


print('== nothing is copied in ==')
base = glove(120)
result = defectsynth.insert_defect(base, stain(), (200, 200), profile_for(120))
check('it produced an image', result['passed'] and result['image'] is not None,
      result['snr'])

if result['passed']:
    out = result['image']
    changed = np.abs(out.astype(np.int16) - base.astype(np.int16)) > 0
    ys, xs = np.nonzero(changed)
    check('only a patch changed, not the whole frame',
          changed.mean() < 0.05, round(float(changed.mean()), 4))
    check('and it is where it was put',
          150 < xs.mean() < 250 and 150 < ys.mean() < 250,
          (round(float(xs.mean())), round(float(ys.mean()))))
    # Far from the insert the image has to be bit-for-bit what arrived.
    far = np.s_[0:100, 0:100]
    check('the rest of the picture is untouched to the bit',
          np.array_equal(out[far], base[far]))

print('\n== the same defect on four glove colours ==')
# A mono camera turns each rubber colour into one grey. These stand for white,
# blue, green and black gloves under the same backlight.
levels = [180, 140, 100, 60]
rows = []
for level in levels:
    frame = glove(level)
    got = defectsynth.insert_defect(frame, stain(), (200, 200), profile_for(level))
    if not got['passed']:
        check(f'grey {level} produced a defect', False, got['snr'])
        continue
    out = got['image'].astype(np.float32) + 1.0
    before = frame.astype(np.float32) + 1.0
    core = np.s_[190:210, 190:210]
    # The same physical defect should change OD by the same amount on every
    # colour, which means it changes grey by different amounts.
    od_shift = float(np.log(before[core]).mean() - np.log(out[core]).mean())
    grey_shift = float(before[core].mean() - out[core].mean())
    rows.append((level, od_shift, grey_shift))
    print(f'    grey {level:3}  OD +{od_shift:.3f}  grey -{grey_shift:5.1f}')

if len(rows) == len(levels):
    shifts = [r[1] for r in rows]
    spread = max(shifts) - min(shifts)
    check('the change in optical density is the same on all four',
          spread < 0.06, [round(s, 3) for s in shifts])
    greys = [r[2] for r in rows]
    check('so the change in grey is not -- it falls with the glove',
          greys[0] > greys[-1] * 1.8, [round(g, 1) for g in greys])
    check('and it is darker after a stain, on every colour',
          all(g > 0 for g in greys), [round(g, 1) for g in greys])

print('\n== a hole is backlight, not thinner rubber ==')
# Absolute, not proportional: the lamp behind the glove is the same brightness
# whatever the glove is made of, so every colour reads the same inside a hole.
insides = []
for level in levels:
    frame = glove(level)
    got = defectsynth.insert_defect(frame, hole(), (200, 200), profile_for(level))
    if not got['passed']:
        check(f'a hole shows on grey {level}', False, got['snr'])
        continue
    insides.append(float(got['image'][196:204, 196:204].mean()))
    print(f'    grey {level:3}  inside the hole {insides[-1]:.1f}')

if len(insides) == len(levels):
    check('every colour reads the same inside the hole',
          max(insides) - min(insides) < 12, [round(v, 1) for v in insides])
    check('and that reading is the backlight',
          abs(np.mean(insides) - BG_LEVEL) < 25, round(float(np.mean(insides)), 1))

print('\n== the box is what can be seen, not what was pasted ==')
boxes = {}
for level in (180, 60):
    frame = glove(level)
    got = defectsynth.insert_defect(frame, stain(strength=0.28), (200, 200),
                                    profile_for(level))
    if got['passed']:
        boxes[level] = got['bbox'][3] * SIZE      # width in pixels
        print(f'    grey {level:3}  box {boxes[level]:.0f}px  snr {got["snr"]}')
    else:
        print(f'    grey {level:3}  refused, snr {got["snr"]}')

if len(boxes) == 2:
    check('the same defect is boxed smaller on the darker glove',
          boxes[60] < boxes[180], boxes)

print('\n== too faint to see is refused, not written ==')
# A defect nobody can see is worse than none: it teaches the detector that a
# patch of ordinary rubber is a tear.
faint = defectsynth.insert_defect(glove(120), stain(strength=0.004),
                                  (200, 200), profile_for(120))
check('it does not pass', not faint['passed'], faint)
check('there is no image to use', faint['image'] is None)
check('and the reason is a number, not a shrug',
      isinstance(faint['snr'], float) and faint['snr'] < defectsynth.SNR_MIN,
      faint['snr'])

strong = defectsynth.insert_defect(glove(120), stain(strength=0.5),
                                   (200, 200), profile_for(120))
check('while a clear one passes the same gate',
      strong['passed'] and strong['snr'] >= defectsynth.SNR_MIN, strong['snr'])

print('\n== a camera with a different scale and a softer lens ==')
# The defect was collected at 0.05 mm per pixel and is being placed on a
# camera at 0.10, so it must come out half as many pixels across -- the same
# physical size, not the same pixel size.
coarse = profile_for(120, mm_per_px=0.10, psf_sigma=2.0)
fine = defectsynth.insert_defect(glove(120), stain(), (200, 200), profile_for(120))
scaled = defectsynth.insert_defect(glove(120), stain(), (200, 200), coarse)
if fine['passed'] and scaled['passed']:
    check('it lands about half the width',
          0.35 < scaled['bbox'][3] / fine['bbox'][3] < 0.75,
          round(scaled['bbox'][3] / fine['bbox'][3], 3))

print('\n== what it refuses to be given ==')
for bad, why in ((np.zeros((10, 10, 3), np.uint8), 'a colour image'),
                 (None, 'nothing at all')):
    try:
        defectsynth.insert_defect(bad, stain(), (5, 5), profile_for(120))
        check(f'{why} is refused', False, 'it was accepted')
    except ValueError:
        check(f'{why} is refused', True)

try:
    odd = stain()
    odd['defect_class'] = 'scratch'
    defectsynth.insert_defect(glove(120), odd, (200, 200), profile_for(120))
    check('an unknown defect class is refused', False, 'it was accepted')
except ValueError:
    check('an unknown defect class is refused', True)

print('\n== placed at the edge of the frame ==')
edge = defectsynth.insert_defect(glove(120), stain(), (62, 62), profile_for(120))
check('it does not crash, whatever it decides', isinstance(edge, dict), edge)

print('\n' + ('DEFECT SYNTH OK' if not fails else f'{len(fails)} FAILED: {fails}'))
sys.exit(1 if fails else 0)
