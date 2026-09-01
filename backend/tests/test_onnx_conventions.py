"""
Finding the conventions a model was built with, on a model whose are known.

An ONNX records the shape of what goes in and nothing about what the numbers
mean: whether the frame is padded to a square or squashed into one, whether
the channels are RGB or BGR, whether pixels arrive as 0..1 or 0..255. Get one
wrong and nothing raises -- the model returns confident nonsense, usually a
single box over the whole picture. That is what a real glove-defect model did:

    class_10   98%   6, 0, 1595, 1199        on a 1600x1200 photo

So probe_onnx.py runs the combinations and reports which ones look like
working detection. Checking that on a real model is circular -- its
conventions are the unknown -- so the model here is built to have known ones.

It finds the blue rectangle in the picture, and it can only do that when fed
BGR at 0..255: read as RGB it looks at the red channel and sees nothing, and
handed 0..1 every pixel falls under its threshold. Squashed or padded it finds
the rectangle either way, but the box lands somewhere different, which is why
the boxes are checked and not just the count.

    python backend/tests/test_onnx_conventions.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_conv_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402
import torch.nn as nn                                   # noqa: E402

from services.onnxrunner import OnnxDetector            # noqa: E402

SIZE = 320
BRIGHT = 200.0            # a threshold only 0..255 input can cross

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


class FindsTheBlueBox(nn.Module):
    """
    Returns the bounding box of the bright pixels in channel 0, normalised.

    Deliberately convention-bound: channel 0 is blue only if it was fed BGR,
    and the threshold is only reachable if the pixels arrived as 0..255.
    """

    def forward(self, x):
        plane = x[0, 0]                                  # [S, S]
        mask = (plane > BRIGHT).to(torch.float32)

        columns = mask.amax(dim=0)                       # any row lit, per column
        rows = mask.amax(dim=1)                          # any column lit, per row
        index = torch.arange(SIZE, dtype=torch.float32)

        # min index where present, max index where present; SIZE and 0 when
        # nothing is lit, which collapses the box and reports no detection.
        x1 = (index * columns + (1 - columns) * SIZE).min()
        x2 = (index * columns).max()
        y1 = (index * rows + (1 - rows) * SIZE).min()
        y2 = (index * rows).max()

        box = torch.stack([x1, y1, x2, y2]) / SIZE
        score = columns.max().reshape(1, 1)
        return (box.reshape(1, 1, 4),
                torch.zeros(1, 1, dtype=torch.int64) + int(columns.sum() > 0) - 1 + 1,
                score)


def photo(cx, cy, w, h, colour=(255, 40, 40)):
    """A dark 640x480 frame with one bright rectangle in it, in BGR."""
    frame = np.full((480, 640, 3), 20, np.uint8)
    cv2.rectangle(frame, (cx - w // 2, cy - h // 2),
                  (cx + w // 2, cy + h // 2), colour, -1)
    return frame


model_path = TMP / 'blue.onnx'
torch.onnx.export(FindsTheBlueBox(), torch.zeros(1, 3, SIZE, SIZE),
                  str(model_path), opset_version=12, input_names=['image_tensor'],
                  output_names=['detected_boxes', 'detected_classes',
                                'detected_scores'], dynamo=False)

# Three frames with the rectangle in three places, so a configuration that
# answers the same thing every time can be told from one that is looking.
frames = [photo(200, 150, 120, 80), photo(430, 320, 90, 140),
          photo(320, 240, 200, 60)]

print('== fed the way it expects ==')
right = OnnxDetector(model_path, display_name='blue.onnx',
                     resize='stretch', channels='bgr', scale='raw')
boxes = [right.predict(f, threshold=0.5) for f in frames]
check('it finds one object in each frame',
      [len(b) for b in boxes] == [1, 1, 1], [len(b) for b in boxes])

if all(boxes):
    found = [b[0]['box'] for b in boxes]
    wanted = [[140, 110, 260, 190], [385, 250, 475, 390], [220, 210, 420, 270]]
    for got, want, name in zip(found, wanted, ('first', 'second', 'third')):
        check(f'the {name} box is where the rectangle is',
              all(abs(a - b) <= 4 for a, b in zip(got, want)), f'{got} wanted {want}')
    check('and the boxes are not the whole frame',
          all((b[2] - b[0]) * (b[3] - b[1]) < 0.5 * 640 * 480 for b in found),
          found)

print('\n== fed the wrong way ==')
for label, kwargs in (
        ('as RGB, so it reads the red channel',
         dict(resize='stretch', channels='rgb', scale='raw')),
        ('as 0..1, so nothing crosses its threshold',
         dict(resize='stretch', channels='bgr', scale='unit')),
):
    wrong = OnnxDetector(model_path, display_name='blue.onnx', **kwargs)
    total = sum(len(wrong.predict(f, threshold=0.5)) for f in frames)
    check(label, total == 0, f'{total} detection(s)')

print('\n== padded instead of squashed ==')
# Both fittings are undone faithfully, so this model -- which only looks for
# bright pixels and does not care about aspect -- comes back right either way.
# Which means the numbers cannot say which fitting a model wants. A real
# detector does care, because it was trained on one of them, but the
# difference shows as slightly worse boxes rather than as no detections, and
# no automatic test tells worse from right. That one is settled by looking at
# a picture, which is what the probe writes out.
padded = OnnxDetector(model_path, display_name='blue.onnx',
                      resize='letterbox', channels='bgr', scale='raw')
padded_boxes = [b['box'] for b in padded.predict(frames[0], threshold=0.5)[:1]]
check('it still finds something', len(padded_boxes) == 1, padded_boxes)
if padded_boxes:
    check('and undoing the padding lands in the same place',
          all(abs(a - b) <= 4 for a, b in zip(padded_boxes[0],
                                              [140, 110, 260, 190])),
          padded_boxes[0])

print('\n== what the probe makes of it ==')
folder = TMP / 'frames'
folder.mkdir()
for index, frame in enumerate(frames):
    cv2.imwrite(str(folder / f'{index}.jpg'), frame)

sys.argv = ['probe', str(model_path), str(folder), '--threshold', '0.5']
sys.path.insert(0, str(REPO / 'backend' / 'tools'))
import probe_onnx                                       # noqa: E402

rows = []
for resize in probe_onnx.RESIZES:
    for channels in probe_onnx.CHANNELS:
        for scale in probe_onnx.SCALES:
            row = probe_onnx.try_one(model_path, [(f'{i}.jpg', f)
                                                  for i, f in enumerate(frames)],
                                     0.5, resize, channels, scale, 'xyxy')
            row['label'] = f'{resize} {channels} {scale}'
            rows.append(row)

working = [r for r in rows
           if probe_onnx.verdict(r, len(frames)).startswith('PLAUSIBLE')]
labels = {r['label'] for r in working}
check('it rules out every configuration fed the wrong channels or range',
      not any('rgb' in l or 'unit' in l for l in labels), sorted(labels))
check('and keeps the one the model actually wants',
      'stretch bgr raw' in labels, sorted(labels))

print('\n== and writes the pictures to settle the rest ==')
# The channels and the pixel range are decided by the numbers. The fitting and
# the box order are not: both give boxes that look reasonable on their own and
# only a picture says which is right. So the probe draws them.
written = probe_onnx.write_previews(
    model_path, [(f'{i}.jpg', f) for i, f in enumerate(frames)],
    0.5, TMP / 'previews', [{'label': 'stretch   bgr raw  xyxy'},
                            {'label': 'stretch   bgr raw  yxyx'}])
check('one image per configuration per frame', len(written) == 6, len(written))
check('and they are readable',
      all(cv2.imread(str(p)) is not None for p in written),
      [p.name for p in written[:2]])

print('\n' + ('ONNX CONVENTIONS OK' if not fails else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
