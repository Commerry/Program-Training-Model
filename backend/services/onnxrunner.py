"""
Run a YOLO-shaped ONNX directly, without ultralytics.

Ultralytics is the better path when it works: it knows the letterboxing, the
class names and the NMS settings a model was exported with. But it reads the
metadata the exporter left in the file and trusts it, and a model built by
other tooling carries whatever that tooling wrote. Seen from one user's export:

    TypeError: empty(): argument 'size' failed to unpack the object at pos 2
               with error "type must be tuple of ints, but got str"

which is ultralytics passing a metadata value straight into torch. Nothing
about the model was wrong; the note attached to it was in the wrong shape. A
detector nobody can run because of a string in a metadata field is a poor
outcome, so this ignores the metadata entirely and works from the graph.

What it understands is the layout every YOLOv8/v9/v10/v11 export shares:
an image in as [1, 3, H, W], and one tensor out of [1, 4 + classes, anchors]
holding box centres, sizes and per-class scores. That covers the exports this
application produces and the ones people arrive with. Anything genuinely
different is reported rather than guessed at.
"""

import numpy as np

from services.projects import ProjectError

# When the graph does not say what size it wants -- a dynamic export -- this is
# what it gets. 640 is what almost every YOLO is trained at, and a detector run
# at the wrong scale detects less rather than failing, so a sensible default
# beats refusing.
FALLBACK_SIZE = 640

DEFAULT_IOU = 0.45


def _letterbox(image, size):
    """
    Fit an image into a square without distorting it, as YOLO training does.

    Returns the padded image and what is needed to put a box back where it
    belongs in the original: the scale it was shrunk by and the padding added.
    Getting this wrong does not fail, it just puts every box in the wrong
    place, which is worse.
    """
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))

    import cv2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def _nms(boxes, scores, threshold):
    """Greedy non-maximum suppression, returning the indices that survive."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size:
        current = order[0]
        keep.append(int(current))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        overlap = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[current] + areas[rest] - overlap
        iou = np.where(union > 0, overlap / np.maximum(union, 1e-9), 0)
        order = rest[iou <= threshold]
    return keep


def describe(path):
    """
    What the graph itself says, ignoring any metadata.

    Returns (input_size, class_count) with either as None when the graph does
    not commit to it.
    """
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    except Exception as exc:  # noqa: BLE001
        raise ProjectError(f'That ONNX could not be opened: {str(exc)[:200]}')

    inputs = session.get_inputs()
    if not inputs:
        raise ProjectError('That ONNX has no input, so nothing can be fed to it.')

    shape = inputs[0].shape
    size = None
    if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
        size = int(shape[2])

    classes = None
    outputs = session.get_outputs()
    if outputs:
        out_shape = outputs[0].shape
        if len(out_shape) == 3 and isinstance(out_shape[1], int):
            classes = int(out_shape[1]) - 4
    return size, classes


class OnnxDetector:
    """A YOLO-shaped ONNX, driven from the graph rather than the metadata."""

    def __init__(self, path, img_size=None, display_name=None):
        # Named after what the person chose, not the temporary file an upload
        # is saved under: an error about a filename they have never seen tells
        # them nothing.
        from pathlib import Path as _Path
        self.name = display_name or _Path(path).name

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                str(path), providers=['CPUExecutionProvider'])
        except Exception as exc:  # noqa: BLE001
            raise ProjectError(
                f'"{self.name}" could not be opened: {str(exc)[:200]}')

        inputs = self.session.get_inputs()
        if not inputs:
            raise ProjectError(
                f'"{self.name}" has no input, so nothing can be fed to it.')
        self.input_name = inputs[0].name

        shape = inputs[0].shape
        if len(shape) != 4:
            raise ProjectError(
                f'"{self.name}" takes an input shaped {shape}. A detector is '
                'driven with a 4-dimensional [batch, channels, height, width] '
                'image.')

        # A fixed size in the graph wins over anything asked for: feeding a
        # static model a different size is an error, not a preference.
        if isinstance(shape[2], int) and shape[2] > 0:
            self.size = int(shape[2])
        else:
            self.size = int(img_size or FALLBACK_SIZE)

        outputs = self.session.get_outputs()
        if not outputs:
            raise ProjectError(f'"{self.name}" produces no output.')
        self.output_name = outputs[0].name

    def predict(self, image_bgr, threshold=0.25, iou=DEFAULT_IOU):
        """
        Detections in the original image's pixels.

        Returns [{class_id, score, box:[x1,y1,x2,y2]}], already suppressed.
        """
        import cv2

        padded, scale, pad_x, pad_y = _letterbox(image_bgr, self.size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        raw = self.session.run([self.output_name], {self.input_name: blob})[0]
        raw = np.asarray(raw)
        if raw.ndim != 3:
            raise ProjectError(
                f'"{self.name}" produces an output shaped {raw.shape}. This '
                'runner understands the [1, 4 + classes, anchors] layout every '
                'YOLO export uses.')

        # [1, C, N] is the usual order; some exports transpose it.
        predictions = raw[0]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T          # -> [anchors, 4 + classes]
        if predictions.shape[1] < 5:
            raise ProjectError(
                f'"{self.name}" gives {predictions.shape[1]} values per anchor, '
                'and a detector needs at least four box numbers and one class '
                'score.')

        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        keep = scores >= threshold
        if not np.any(keep):
            return []
        boxes_cxcywh = boxes_cxcywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        cx, cy, w, h = (boxes_cxcywh[:, 0], boxes_cxcywh[:, 1],
                        boxes_cxcywh[:, 2], boxes_cxcywh[:, 3])
        corners = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        # Back out of the letterbox into the original frame.
        corners[:, [0, 2]] -= pad_x
        corners[:, [1, 3]] -= pad_y
        corners /= max(scale, 1e-9)

        height, width = image_bgr.shape[:2]
        corners[:, [0, 2]] = corners[:, [0, 2]].clip(0, width)
        corners[:, [1, 3]] = corners[:, [1, 3]].clip(0, height)

        # Suppression runs per class, which is what ultralytics does. Across
        # classes it would drop the second of two objects that genuinely
        # overlap -- a label on a bottle, a digit inside a plate -- and keep
        # only whichever the model happened to score higher.
        keep_indices = []
        for class_id in np.unique(class_ids):
            members = np.flatnonzero(class_ids == class_id)
            survivors = _nms(corners[members], scores[members], iou)
            keep_indices.extend(int(members[i]) for i in survivors)
        keep_indices.sort(key=lambda i: -scores[i])

        detections = []
        for index in keep_indices:
            x1, y1, x2, y2 = (int(round(v)) for v in corners[index])
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            detections.append({
                'class_id': int(class_ids[index]),
                'score': float(scores[index]),
                'box': [x1, y1, x2, y2],
            })
        return detections
