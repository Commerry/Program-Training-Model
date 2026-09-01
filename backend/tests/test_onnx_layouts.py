"""
Every output layout a detector ONNX arrives in, with a box in a known place.

There is no single YOLO output layout. A model built elsewhere may hand back
one fused tensor, a table of rows with NMS already applied, four tensors of
counts and boxes and scores and labels, or the un-fused head a Luxonis blob is
built from. Reading only the first output and assuming it is the fused one
gave a user

    "model.onnx" gives 1 values per anchor

on a model that was perfectly fine.

The point here is not that each file runs. It is that the box comes back where
it was put: a decoder that reads the columns in the wrong order still returns
detections, and they are all in the wrong place. So every layout carries one
object at a known position, and the position is what is checked.

Three of the five make ultralytics raise, which is how the direct runner comes
to be used at all, so each is also driven through the endpoint.

    python backend/tests/test_onnx_layouts.py
"""
import io
import os
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_layout_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['REQUIRE_AUTH'] = '0'
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402
import torch.nn as nn                                   # noqa: E402
from app import create_app                              # noqa: E402

c = create_app().test_client()

SIZE = 640
# A 640x480 photo letterboxes into 640 at scale 1.0 with 80px bars top and
# bottom, so a box at [100, 180, 200, 280] in the network's view belongs at
# [100, 100, 200, 200] in the photo.
BOX_IN = [100.0, 180.0, 200.0, 280.0]
BOX_OUT = [100, 100, 200, 200]
CLASS = 1
SCORE = 0.9

photo = np.full((480, 640, 3), 50, np.uint8)
cv2.rectangle(photo, (100, 100), (200, 200), (230, 235, 245), -1)
JPEG = cv2.imencode('.jpg', photo)[1].tobytes()

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


class Emits(nn.Module):
    """Returns fixed tensors, with a convolution so there is a graph at all."""

    def __init__(self, tensors, names):
        super().__init__()
        self.stem = nn.Conv2d(3, 2, 3, 2, 1)
        self.names = names
        for i, t in enumerate(tensors):
            self.register_buffer(f'out{i}', t)
        self.count = len(tensors)

    def forward(self, x):
        # The outputs are fixed, but they have to depend on the input or the
        # exporter prunes the whole graph and the model ends up with no input.
        nothing = self.stem(x).sum() * 0.0
        return tuple(getattr(self, f'out{i}') + nothing.to(getattr(self, f'out{i}').dtype)
                     for i in range(self.count))


def export(tensors, names, path):
    torch.onnx.export(Emits(tensors, names), torch.zeros(1, 3, SIZE, SIZE),
                      str(path), opset_version=12, input_names=['images'],
                      output_names=names, dynamo=False)
    return path


def fused():
    """[1, 4 + classes, anchors] -- what ultralytics exports by default."""
    anchors, classes = 8400, 3
    out = np.zeros((1, 4 + classes, anchors), np.float32)
    cx = (BOX_IN[0] + BOX_IN[2]) / 2
    cy = (BOX_IN[1] + BOX_IN[3]) / 2
    out[0, :4, 0] = [cx, cy, BOX_IN[2] - BOX_IN[0], BOX_IN[3] - BOX_IN[1]]
    out[0, 4 + CLASS, 0] = SCORE
    return [torch.from_numpy(out)], ['output0']


def decoded():
    """[1, n, 6] of x1 y1 x2 y2 score class -- exported with nms=True."""
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0, :4] = BOX_IN
    out[0, 0, 4] = SCORE
    out[0, 0, 5] = CLASS
    return [torch.from_numpy(out)], ['output0']


def batched():
    """[n, 7] of batch x1 y1 x2 y2 class score -- YOLOv5/v7 end2end."""
    out = np.zeros((4, 7), np.float32)
    out[0] = [0, BOX_IN[0], BOX_IN[1], BOX_IN[2], BOX_IN[3], CLASS, SCORE]
    return [torch.from_numpy(out)], ['output0']


def split():
    """num_dets / boxes / scores / labels -- the EfficientNMS layout."""
    n = 100
    boxes = np.zeros((1, n, 4), np.float32)
    boxes[0, 0] = BOX_IN
    scores = np.zeros((1, n), np.float32)
    scores[0, 0] = SCORE
    labels = np.zeros((1, n), np.int32)
    labels[0, 0] = CLASS
    count = np.array([[1]], np.int32)
    return ([torch.from_numpy(count), torch.from_numpy(boxes),
             torch.from_numpy(scores), torch.from_numpy(labels)],
            ['num_dets', 'boxes', 'scores', 'labels'])


def split_no_count():
    """
    detected_boxes / detected_classes / detected_scores, and no count.

    A real model arrived shaped exactly like this, with `num` dynamic and the
    boxes as fractions of the input rather than pixels. On a frame holding one
    object every tensor here holds a single value, so a decoder looking for a
    num_dets among them finds the data instead -- which is how this one came
    back as "it returned (1, 1, 4), (1, 1), (1, 1)".
    """
    boxes = np.array([[[BOX_IN[0] / SIZE, BOX_IN[1] / SIZE,
                        BOX_IN[2] / SIZE, BOX_IN[3] / SIZE]]], np.float32)
    classes = np.array([[CLASS]], np.int64)
    scores = np.array([[SCORE]], np.float32)
    return ([torch.from_numpy(boxes), torch.from_numpy(classes),
             torch.from_numpy(scores)],
            ['detected_boxes', 'detected_classes', 'detected_scores'])


def boxes_and_scores():
    """[1, n, 4] beside [1, n, classes], with no suppression applied."""
    n, classes = 200, 4
    boxes = np.zeros((1, n, 4), np.float32)
    matrix = np.zeros((1, n, classes), np.float32)
    # The same object twice over, so the suppression this layout needs shows.
    for row, nudge in ((0, 0.0), (1, 2.0)):
        boxes[0, row] = [BOX_IN[0] + nudge, BOX_IN[1] + nudge,
                         BOX_IN[2] + nudge, BOX_IN[3] + nudge]
        matrix[0, row, CLASS] = SCORE - row * 0.05
    return [torch.from_numpy(boxes), torch.from_numpy(matrix)], ['boxes', 'scores']


def strides():
    """
    Three [1, 64 + classes, h, w] head outputs -- the un-fused export.

    The four box sides are distributions over sixteen bins, so a one-hot at
    bin 1 on a stride-32 map means a distance of 32 pixels on that side.
    """
    classes, tensors = 3, []
    for grid in (80, 40, 20):
        stride = SIZE / grid
        feature = np.zeros((1, 64 + classes, grid, grid), np.float32)
        if grid == 20:
            # anchor (4, 5) sits at (144, 176); one bin out on every side.
            for side in range(4):
                feature[0, side * 16 + 1, 5, 4] = 20.0
        feature[0, 64:] -= 20.0                    # background, sigmoid ~ 0
        if grid == 20:
            # logit chosen so the sigmoid lands on the same 0.9 as the rest.
            feature[0, 64 + 2, 5, 4] = float(np.log(SCORE / (1 - SCORE)))
        tensors.append(torch.from_numpy(feature))
    return tensors, ['s8', 's16', 's32']


def run(name, builder, expect_box=None, expect_class=None, expect_layout=None):
    """
    The runner is driven directly here, not through the endpoint.

    Ultralytics loads these files -- there is no metadata in them to choke on
    -- so going through the endpoint would test ultralytics, which is not what
    is in question. What ultralytics then does with them is checked separately
    below, and it is not good.
    """
    from services.onnxrunner import OnnxDetector

    tensors, names = builder()
    path = export(tensors, names, TMP / f'{name}.onnx')

    print(f'\n== {name} ==')
    try:
        detector = OnnxDetector(path, display_name='model.onnx')
        found = detector.predict(photo, threshold=0.5)
    except Exception as exc:  # noqa: BLE001
        check('it runs', False, f'{type(exc).__name__}: {str(exc)[:200]}')
        return
    check('it runs', True)
    check(f'read as the {expect_layout} layout',
          detector.layout == expect_layout, detector.layout)
    check('it found the object', len(found) == 1, len(found))
    if not found:
        return
    box = found[0]['box']
    check('and put it where it belongs',
          all(abs(a - b) <= 3 for a, b in zip(box, expect_box)),
          f'{box} wanted {expect_box}')
    check('with the right class',
          found[0]['class_id'] == expect_class, found[0]['class_id'])
    check('and the right confidence',
          abs(found[0]['score'] - SCORE) < 0.02, found[0]['score'])

    # What ultralytics makes of the same file, since it is tried first.
    from ultralytics import YOLO
    try:
        theirs = YOLO(str(path))(photo, conf=0.5, verbose=False)
        count = len(theirs[0].boxes)
    except Exception as exc:  # noqa: BLE001
        count = f'{type(exc).__name__}'
    print(f'    (ultralytics on the same file: {count})')

    # And through the endpoint, which tries ultralytics first: whatever it
    # does, a detection has to come back.
    r = c.post('/api/models/test', data={
        'images': [(io.BytesIO(JPEG), 'a.jpg')],
        'model': (io.BytesIO(path.read_bytes()), 'model.onnx'),
        'score_threshold': '0.5',
    }, content_type='multipart/form-data')
    body = r.get_json() or {}
    results = body.get('results') or []
    through = results[0].get('detections') if results else []
    check('the endpoint finds it too',
          r.status_code == 200 and len(through or []) == 1,
          f'{r.status_code} {str(body.get("message") or body.get("device"))[:90]} '
          f'{len(through or [])} found')
    print(f'    (endpoint used: {body.get("device")})')


run('fused', fused, BOX_OUT, CLASS, 'fused')
run('decoded', decoded, BOX_OUT, CLASS, 'decoded')
run('batched', batched, BOX_OUT, CLASS, 'decoded')
run('split', split, BOX_OUT, CLASS, 'split')
# Those three output names are Azure Custom Vision's, so this one is
# recognised as that export and fed the way it wants -- squashed into the
# square rather than padded -- which puts the box back through different
# arithmetic. 640x480 squashed into 640 stretches the vertical by 4/3, so the
# same normalised box lands at y 135..210 instead of 100..200.
run('split with no count', split_no_count, [100, 135, 200, 210], CLASS, 'split')
run('boxes and scores', boxes_and_scores, BOX_OUT, CLASS, 'boxes and scores')
# anchor (144,176) with 32px on every side -> [112,144,176,208] in the
# network's view, which is [112, 64, 176, 128] in the photo.
run('strides', strides, [112, 64, 176, 128], 2, 'per-stride')

print('\n' + ('ONNX LAYOUTS OK' if not fails else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    import shutil
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
