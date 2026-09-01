"""
Run a detector ONNX directly, without ultralytics.

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

There is no single YOLO output layout, which was the next thing to go wrong
here. Reading only the first output and assuming it is the fused one gives

    "model.onnx" gives 1 values per anchor

on any export that splits its results across several tensors. So every output
is read, and the shape of what actually comes back decides how to decode it:

  fused       one tensor, [1, 4+classes, anchors] -- what this app exports,
              and what ultralytics produces by default
  decoded     [1, n, 6] of x1 y1 x2 y2 score class -- an export with NMS
              baked in (ultralytics nms=True)
  batched     [n, 7] of batch x1 y1 x2 y2 class score -- YOLOv5/v7 end2end
  split       num_dets / boxes / scores / labels across four tensors -- the
              TensorRT EfficientNMS layout, and what most edge toolchains emit
  per-stride  three [1, channels, h, w] head outputs -- the un-fused export a
              Luxonis blob is built from

Anything else is reported with the whole graph signature attached, because a
shape nobody can name is a thing to be told about, not guessed at.
"""

import numpy as np

from services.projects import ProjectError

# When the graph does not say what size it wants -- a dynamic export -- this is
# what it gets. 640 is what almost every YOLO is trained at, and a detector run
# at the wrong scale detects less rather than failing, so a sensible default
# beats refusing.
FALLBACK_SIZE = 640

DEFAULT_IOU = 0.45

# Box channels in a YOLOv8/v11 head: four sides, each a distribution over
# sixteen bins. Fixed across every ultralytics model of that generation.
DFL_BINS = 16
BOX_CHANNELS = 4 * DFL_BINS


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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _looks_like_class_column(column):
    """
    Whole numbers, none of them negative, and not very many distinct ones.

    This is what separates an already-decoded [n, 6] of boxes and class ids
    from a fused [1, 6, anchors] of a two-class model, which has the same
    trailing dimension and would otherwise be decoded as the wrong thing.
    """
    if column.size == 0:
        return False
    finite = column[np.isfinite(column)]
    if finite.size == 0:
        return False
    return bool(np.all(np.abs(finite - np.round(finite)) < 1e-3)
                and np.all(finite >= -0.5) and np.max(finite) < 4096)


def _looks_like_score_column(column):
    if column.size == 0:
        return False
    finite = column[np.isfinite(column)]
    return bool(finite.size and finite.min() >= -1e-3 and finite.max() <= 1.001)


def _looks_like_corners(columns):
    """
    Four columns that are x1 y1 x2 y2 rather than centre-x centre-y w h.

    The strongest signal available, and the one that settles a two-class model
    whose class ids of 0 and 1 look like scores: corners always have x1 <= x2,
    while a centre and a width almost never do -- an object sitting anywhere
    but hard against the left edge has a centre larger than its own width.
    """
    if columns.shape[0] == 0:
        return False
    finite = columns[np.all(np.isfinite(columns), axis=1)]
    if finite.shape[0] == 0:
        return False
    return bool(np.all(finite[:, 0] <= finite[:, 2] + 1e-3)
                and np.all(finite[:, 1] <= finite[:, 3] + 1e-3))


RESIZES = ('letterbox', 'stretch')
CHANNELS = ('rgb', 'bgr')
SCALES = ('unit', 'raw')
ORDERS = ('xyxy', 'yxyx')


def parse_conventions(text):
    """
    Read "stretch bgr raw xyxy" into keyword arguments, in any order.

    What probe_onnx.py prints is what this accepts, so the answer it works out
    can be pasted straight in. Anything unrecognised is ignored rather than
    refused: a typo should cost a default, not a run.
    """
    chosen = {}
    for word in str(text or '').replace(',', ' ').lower().split():
        if word in RESIZES:
            chosen['resize'] = word
        elif word in CHANNELS:
            chosen['channels'] = word
        elif word in SCALES:
            chosen['scale'] = word
        elif word in ORDERS:
            chosen['box_order'] = word
    return chosen


# Azure Custom Vision exports a detector as these three outputs, and its own
# sample code feeds it in a way no YOLO is fed: the frame resized to the square
# without preserving aspect, channels swapped to BGR, and pixels left at
# 0..255 rather than divided down. Fed the YOLO way it does not fail -- it
# answers with one box over the whole picture, which is how a real one behaved
# here. The names are how the export is recognised; a folder from that export
# also carries cvexport.manifest, labels.txt and metadata_properties.json.
CUSTOM_VISION_OUTPUTS = {'detected_boxes', 'detected_classes', 'detected_scores'}
CUSTOM_VISION_CONVENTIONS = {'resize': 'stretch', 'channels': 'bgr',
                             'scale': 'raw', 'box_order': 'xyxy'}


def _is_custom_vision(session):
    names = {node.name for node in session.get_outputs()}
    return CUSTOM_VISION_OUTPUTS.issubset(names)


def is_custom_vision_file(path):
    """Recognise the export before anything tries to load it as a YOLO."""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(path),
                                       providers=['CPUExecutionProvider'])
    except Exception:  # noqa: BLE001 - the caller reports load failures
        return False
    return _is_custom_vision(session) or bool(sidecars(path).get('is_custom_vision'))


def sidecars(model_path):
    """
    What an export folder says about itself, when the folder is to hand.

    Custom Vision writes labels.txt and metadata_properties.json beside the
    model, and between them they hold both the class names and the
    preprocessing -- the two things the ONNX itself does not record. Reading
    them beats inferring them.
    """
    from pathlib import Path as _Path
    folder = _Path(model_path).parent
    found = {}

    labels = folder / 'labels.txt'
    if labels.is_file():
        try:
            names = [line.strip() for line in
                     labels.read_text(encoding='utf-8-sig').splitlines()]
            found['labels'] = [n for n in names if n]
        except OSError:
            pass

    meta = folder / 'metadata_properties.json'
    if meta.is_file():
        try:
            import json
            found['metadata'] = json.loads(meta.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            pass

    found['is_custom_vision'] = (folder / 'cvexport.manifest').is_file()
    return found


def _signature(session):
    """The graph in one line, for an error somebody has to act on."""
    def describe_one(node):
        shape = [d if isinstance(d, int) else str(d) for d in (node.shape or [])]
        return f'{node.name}{shape}'
    ins = ', '.join(describe_one(n) for n in session.get_inputs())
    outs = ', '.join(describe_one(n) for n in session.get_outputs())
    return f'in: {ins or "none"}; out: {outs or "none"}'


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
    for node in session.get_outputs():
        out_shape = node.shape
        # The fused layout is the only one whose class count can be read off
        # the graph; the others carry it in values, not dimensions.
        if len(out_shape) == 3 and all(isinstance(d, int) for d in out_shape[1:]):
            short, long = sorted(out_shape[1:])
            if 5 <= short < long:
                classes = int(short) - 4
                break
    return size, classes


class OnnxDetector:
    """A detector ONNX, driven from the graph rather than the metadata."""

    def __init__(self, path, img_size=None, display_name=None,
                 resize=None, channels=None, scale=None, box_order=None):
        """
        `resize`, `channels`, `scale` and `box_order` are the conventions a
        model was built with, and nothing in an ONNX records them.

        A YOLO is letterboxed, fed RGB in 0..1, and answers in x1 y1 x2 y2, so
        those are the defaults. A model converted from other tooling may be
        stretched to fit rather than padded, may want BGR, may want 0..255,
        and may answer in y1 x1 y2 x2 -- and getting any of them wrong does
        not fail. It returns confident nonsense, usually one box over the
        whole frame, which is how this came up: a real model answered

            class_10  98%  6, 0, 1595, 1199

        on a 1600x1200 photo, which is the entire picture.

        There is no way to read these off the file, so backend/tools/
        probe_onnx.py runs the combinations over real images and reports which
        one produces detections that vary from picture to picture instead of
        covering everything.
        """
        # Named after what the person chose, not the temporary file an upload
        # is saved under: an error about a filename they have never seen tells
        # them nothing.
        from pathlib import Path as _Path
        self.name = display_name or _Path(path).name
        # Left as None until something says otherwise, so that a convention
        # somebody chose can be told apart from one that was never mentioned.
        # Passing channels='rgb' is a decision; not passing it is not, and
        # recognising the export must only fill in the second kind.
        self.resize = resize
        self.channels = channels
        self.scale = scale
        self.box_order = box_order
        self.chosen = {'resize': resize is not None,
                       'channels': channels is not None,
                       'scale': scale is not None,
                       'box_order': box_order is not None}
        self.source = None

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                str(path), providers=['CPUExecutionProvider'])
        except Exception as exc:  # noqa: BLE001
            raise ProjectError(
                f'"{self.name}" could not be opened: {str(exc)[:200]}')

        self.signature = _signature(self.session)

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
                f'image. Graph -- {self.signature}')

        # A fixed size in the graph wins over anything asked for: feeding a
        # static model a different size is an error, not a preference.
        if isinstance(shape[2], int) and shape[2] > 0:
            self.size = int(shape[2])
        else:
            self.size = int(img_size or FALLBACK_SIZE)

        if not self.session.get_outputs():
            raise ProjectError(
                f'"{self.name}" produces no output. Graph -- {self.signature}')

        # A Custom Vision export is recognisable and is fed nothing like a
        # YOLO, so it gets its own conventions rather than the YOLO defaults.
        # Anything the caller stated stands: this fills gaps, it does not
        # overrule.
        if _is_custom_vision(self.session):
            self.source = 'Azure Custom Vision'
            for key, value in CUSTOM_VISION_CONVENTIONS.items():
                if not self.chosen[key]:
                    setattr(self, key, value)

        # The folder an export came from says more than the file does. It is
        # only there when the model is read in place -- an upload arrives
        # alone -- so this is a bonus, not a requirement.
        self.sidecars = sidecars(path)
        self.labels = self.sidecars.get('labels')
        if self.sidecars.get('is_custom_vision'):
            self.source = 'Azure Custom Vision'
        self._apply_declared_preprocessing()

        # Whatever is still unstated falls back to what a YOLO wants, which is
        # what this application exports and much of what arrives here.
        for key, value in (('resize', 'letterbox'), ('channels', 'rgb'),
                           ('scale', 'unit'), ('box_order', 'xyxy')):
            if getattr(self, key) is None:
                setattr(self, key, value)

        # Which decoder was used, once something has actually been run. Worth
        # reporting: a person testing a model from elsewhere wants to know how
        # it was read, not only that it was.
        self.layout = None

    def _apply_declared_preprocessing(self):
        """
        Believe the export's own metadata file over any inference.

        Custom Vision writes what it did in metadata_properties.json --
        the target size, the resize method, and a mean and standard deviation
        the pixels were normalised by. Those are facts about the model, so
        they win over the defaults; anything the caller asked for still wins
        over them.

        A mean and deviation other than the identity is not something this
        runner applies, so it is reported rather than quietly ignored: a model
        fed unnormalised pixels answers confidently and wrongly, which is the
        failure this whole path exists to stop.
        """
        self.notes = []
        meta = (self.sidecars or {}).get('metadata') or {}
        if not meta:
            return

        def number_list(key):
            raw = meta.get(key)
            if not raw:
                return None
            try:
                import json
                value = json.loads(raw) if isinstance(raw, str) else raw
                return [float(v) for v in value]
            except (ValueError, TypeError):
                return None

        width = meta.get('CustomVision.Preprocess.TargetWidth')
        height = meta.get('CustomVision.Preprocess.TargetHeight')
        try:
            if width and height and int(width) == int(height) and int(width) > 0:
                # Only when the graph left the size open; a fixed input wins.
                if not isinstance(self.session.get_inputs()[0].shape[2], int):
                    self.size = int(width)
        except (TypeError, ValueError):
            pass

        method = str(meta.get('CustomVision.Preprocess.ResizeMethod') or '')
        if method and not self.chosen['resize']:
            # Every resize method Custom Vision names squashes the frame into
            # the square; none of them pads it the way a YOLO does.
            self.resize = 'stretch'

        mean = number_list('CustomVision.Preprocess.NormalizeMean')
        std = number_list('CustomVision.Preprocess.NormalizeStd')
        if mean and any(abs(v) > 1e-6 for v in mean):
            self.notes.append(
                f'the export declares a normalisation mean of {mean}, which '
                'this runner does not apply')
        if std and any(abs(v - 1.0) > 1e-6 for v in std):
            self.notes.append(
                f'the export declares a normalisation deviation of {std}, '
                'which this runner does not apply')

    def _unreadable(self, detail):
        return ProjectError(
            f'"{self.name}" produces output this runner cannot read: {detail} '
            f'Graph -- {self.signature}')

    # -- decoders ---------------------------------------------------------
    # Each returns (corners, scores, class_ids, needs_nms), with the corners
    # as x1 y1 x2 y2 in the letterboxed input image. Mapping back to the
    # original frame is the same for all of them and happens once, below.

    def _decode_fused(self, raw):
        """[1, 4 + classes, anchors]: centres, sizes and per-class scores."""
        predictions = raw[0]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T          # -> [anchors, 4 + classes]

        boxes = predictions[:, :4]
        class_scores = predictions[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        corners = np.stack([cx - w / 2, cy - h / 2,
                            cx + w / 2, cy + h / 2], axis=1)
        return corners, scores, class_ids, True

    def _decode_rows(self, rows, order):
        """
        An export with NMS already applied, one row per surviving detection.

        `order` is 'xyxy_score_class' for the six-column ultralytics layout or
        'batch_xyxy_class_score' for the seven-column YOLOv5/v7 one.
        """
        if order == 'xyxy_score_class':
            corners = rows[:, :4].astype(np.float32)
            scores = rows[:, 4].astype(np.float32)
            class_ids = rows[:, 5].astype(np.int32)
        else:
            corners = rows[:, 1:5].astype(np.float32)
            class_ids = rows[:, 5].astype(np.int32)
            scores = rows[:, 6].astype(np.float32)

        # Some exports hand back fractions of the input rather than pixels.
        if corners.size and np.nanmax(corners) <= 1.5:
            corners = corners * self.size
        return corners, scores, class_ids, False

    def _decode_split(self, arrays):
        """
        Boxes, scores and classes as separate tensors, NMS already applied.

        The layout most edge toolchains emit. Some add a num_dets saying how
        many of a fixed number of slots are real; some leave the count as a
        dynamic dimension and return exactly as many rows as there are
        detections. One model arrived as

            detected_boxes[1, num, 4], detected_classes[1, num],
            detected_scores[1, num]

        with no count at all, and on a frame holding a single object num is 1,
        which makes every tensor here a single value and the count -- if one
        were looked for -- indistinguishable from the data. So the count is
        only read when the shapes leave no doubt about which tensor it is.

        Matched on shape rather than name, because the names differ between
        every toolchain that emits this.
        """
        boxes = boxes_index = None
        for index, array in enumerate(arrays):
            flat = array.reshape(-1, 4) if array.size and array.shape[-1] == 4 \
                and array.ndim in (2, 3) else None
            if flat is not None and boxes is None:
                boxes, boxes_index = flat.astype(np.float32), index

        if boxes is None:
            return None
        rows = boxes.shape[0]

        # Anything holding one value per box. Reshaped rather than squeezed:
        # a single detection makes a [1, 1] tensor squeeze down to a scalar,
        # and that vector is exactly what is wanted.
        vectors, spare = [], []
        for index, array in enumerate(arrays):
            if index == boxes_index:
                continue
            if array.size == rows:
                vectors.append(array.reshape(-1))
            else:
                spare.append(array)
        if not vectors:
            return None

        # A num_dets is only recognisable when it cannot be one of the above.
        count = None
        for array in spare:
            if array.size == 1:
                count = int(np.asarray(array).reshape(-1)[0])
                break

        # One vector is the scores and one the class ids. An integer dtype
        # settles it outright, which is how most exporters write the classes.
        # Failing that, whole numbers mark the labels -- except with two
        # classes, where ids of 0 and 1 pass for confidences, so the tie goes
        # to whichever vector carries more distinct values.
        scores = labels = None
        if len(vectors) >= 2:
            first, second = vectors[0], vectors[1]
            first_int = np.issubdtype(first.dtype, np.integer)
            second_int = np.issubdtype(second.dtype, np.integer)
            if first_int != second_int:
                labels, scores = (first, second) if first_int else (second, first)
            else:
                first_whole = _looks_like_class_column(first)
                second_whole = _looks_like_class_column(second)
                if first_whole != second_whole:
                    labels, scores = ((first, second) if first_whole
                                      else (second, first))
                elif len(np.unique(first)) >= len(np.unique(second)):
                    scores, labels = first, second
                else:
                    scores, labels = second, first
        else:
            scores = vectors[0]

        scores = np.asarray(scores, np.float32)
        labels = (np.zeros(len(scores), np.int32) if labels is None
                  else np.asarray(labels).astype(np.int32))

        if count is not None and 0 <= count <= len(scores):
            boxes, scores, labels = boxes[:count], scores[:count], labels[:count]

        # Some of these hand back fractions of the input rather than pixels.
        if boxes.size and np.nanmax(np.abs(boxes)) <= 1.5:
            boxes = boxes * self.size
        return boxes, scores, labels, False

    def _decode_pair(self, arrays):
        """
        Boxes in one tensor, a score per class in another.

        [1, n, 4] alongside [1, n, classes]: no NMS applied, which is what
        separates this from the split layout above, so the boxes go through
        suppression like a fused model's.
        """
        boxes = matrix = None
        for array in arrays:
            if array.ndim != 3 or array.shape[0] != 1:
                continue
            if array.shape[2] == 4 and boxes is None:
                boxes = array[0].astype(np.float32)
            elif array.shape[2] >= 1 and matrix is None:
                matrix = array[0].astype(np.float32)

        if boxes is None or matrix is None or matrix.shape[0] != boxes.shape[0]:
            return None

        class_ids = matrix.argmax(axis=1)
        scores = matrix[np.arange(len(class_ids)), class_ids]
        if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
            scores = _sigmoid(scores)

        if boxes.size and np.nanmax(np.abs(boxes)) <= 1.5:
            boxes = boxes * self.size
        if not _looks_like_corners(boxes):
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2,
                              cx + w / 2, cy + h / 2], axis=1)
        return boxes, scores, class_ids.astype(np.int32), True

    def _decode_strides(self, arrays):
        """
        The un-fused head: one [1, channels, h, w] tensor per stride.

        This is what a Luxonis blob is built from, and what several converters
        leave in place. The four box sides are distributions over sixteen bins
        that have to be averaged back into distances -- ultralytics does this
        inside the model when it exports fused, and not when it does not.
        """
        maps = [a for a in arrays if a.ndim == 4 and a.shape[0] == 1]
        if not maps:
            return None

        all_corners, all_scores, all_ids = [], [], []
        for feature in maps:
            _, channels, height, width = feature.shape
            classes = channels - BOX_CHANNELS
            if classes < 1 or height < 1 or width < 1:
                return None
            stride = self.size / float(height)

            flat = feature[0].reshape(channels, height * width).astype(np.float32)
            distribution = flat[:BOX_CHANNELS].reshape(4, DFL_BINS, -1)
            distribution = distribution - distribution.max(axis=1, keepdims=True)
            weights = np.exp(distribution)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
            bins = np.arange(DFL_BINS, dtype=np.float32).reshape(1, DFL_BINS, 1)
            distances = (weights * bins).sum(axis=1) * stride

            class_map = flat[BOX_CHANNELS:]
            # Exports differ on whether the sigmoid is inside the graph.
            if class_map.min() < 0.0 or class_map.max() > 1.0:
                class_map = _sigmoid(class_map)

            grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                         np.arange(height, dtype=np.float32))
            anchor_x = (grid_x.reshape(-1) + 0.5) * stride
            anchor_y = (grid_y.reshape(-1) + 0.5) * stride
            left, top, right, bottom = distances

            all_corners.append(np.stack([anchor_x - left, anchor_y - top,
                                         anchor_x + right, anchor_y + bottom],
                                        axis=1))
            ids = class_map.argmax(axis=0)
            all_scores.append(class_map[ids, np.arange(class_map.shape[1])])
            all_ids.append(ids)

        return (np.concatenate(all_corners), np.concatenate(all_scores),
                np.concatenate(all_ids).astype(np.int32), True)

    # -- dispatch ---------------------------------------------------------

    def _row_layout(self, rows):
        """
        Is this already-decoded detections, and in which column order?

        A fused two-class model also has six trailing values, so the decision
        is made on what the values look like rather than on the shape alone.
        """
        if rows.shape[1] == 6:
            if _looks_like_score_column(rows[:, 4]) \
                    and _looks_like_class_column(rows[:, 5]) \
                    and _looks_like_corners(rows[:, :4]):
                return 'xyxy_score_class'
            return None
        if rows.shape[1] == 7:
            if _looks_like_score_column(rows[:, 6]) \
                    and _looks_like_class_column(rows[:, 5]) \
                    and _looks_like_corners(rows[:, 1:5]):
                return 'batch_xyxy_class_score'
        return None

    def _decode(self, arrays):
        """Work out which of the known layouts came back, and decode it."""
        if len(arrays) >= 2:
            split = self._decode_split(arrays)
            if split is not None:
                self.layout = 'split'
                return split

            pair = self._decode_pair(arrays)
            if pair is not None:
                self.layout = 'boxes and scores'
                return pair

            strides = self._decode_strides(arrays)
            if strides is not None:
                self.layout = 'per-stride'
                return strides

        for array in arrays:
            rows = None
            if array.ndim == 3 and array.shape[0] == 1 and array.shape[2] in (6, 7) \
                    and array.shape[1] >= array.shape[2]:
                rows = array[0]
            elif array.ndim == 2 and array.shape[1] in (6, 7):
                rows = array
            if rows is None:
                continue
            order = self._row_layout(rows)
            if order:
                self.layout = 'decoded'
                return self._decode_rows(rows, order)

        for array in arrays:
            if array.ndim == 3 and array.shape[0] == 1:
                short, long = sorted(array.shape[1:])
                if short >= 5 and long > short:
                    self.layout = 'fused'
                    return self._decode_fused(array)

        strides = self._decode_strides(arrays)
        if strides is not None:
            self.layout = 'per-stride'
            return strides

        shapes = ', '.join(str(tuple(a.shape)) for a in arrays)
        raise self._unreadable(
            f'it returned {shapes}, and none of those is a layout this runner '
            'knows how to read.')

    def predict(self, image_bgr, threshold=0.25, iou=DEFAULT_IOU):
        """
        Detections in the original image's pixels.

        Returns [{class_id, score, box:[x1,y1,x2,y2]}], already suppressed.
        """
        import cv2

        height, width = image_bgr.shape[:2]
        if self.resize == 'stretch':
            # No padding: the whole frame is squashed into the square, so the
            # two axes come back at different scales.
            fitted = cv2.resize(image_bgr, (self.size, self.size),
                                interpolation=cv2.INTER_LINEAR)
            scale_x, scale_y = self.size / width, self.size / height
            pad_x = pad_y = 0
        else:
            fitted, scale, pad_x, pad_y = _letterbox(image_bgr, self.size)
            scale_x = scale_y = scale

        prepared = (fitted if self.channels == 'bgr'
                    else cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB))
        blob = prepared.transpose(2, 0, 1)[None].astype(np.float32)
        if self.scale == 'unit':
            blob /= 255.0

        raw = self.session.run(None, {self.input_name: blob})
        arrays = [np.asarray(a) for a in raw]
        corners, scores, class_ids, needs_nms = self._decode(arrays)

        corners = np.asarray(corners, np.float32).reshape(-1, 4)
        scores = np.asarray(scores, np.float32).reshape(-1)
        class_ids = np.asarray(class_ids, np.int32).reshape(-1)

        if self.box_order == 'yxyx':
            corners = corners[:, [1, 0, 3, 2]]

        keep = scores >= threshold
        if not np.any(keep):
            return []
        corners, scores, class_ids = corners[keep], scores[keep], class_ids[keep]

        # Back out of the fitting into the original frame.
        corners[:, [0, 2]] -= pad_x
        corners[:, [1, 3]] -= pad_y
        corners[:, [0, 2]] /= max(scale_x, 1e-9)
        corners[:, [1, 3]] /= max(scale_y, 1e-9)

        corners[:, [0, 2]] = corners[:, [0, 2]].clip(0, width)
        corners[:, [1, 3]] = corners[:, [1, 3]].clip(0, height)

        if needs_nms:
            # Suppression runs per class, which is what ultralytics does.
            # Across classes it would drop the second of two objects that
            # genuinely overlap -- a label on a bottle, a digit inside a
            # plate -- and keep only whichever the model scored higher.
            keep_indices = []
            for class_id in np.unique(class_ids):
                members = np.flatnonzero(class_ids == class_id)
                survivors = _nms(corners[members], scores[members], iou)
                keep_indices.extend(int(members[i]) for i in survivors)
            keep_indices.sort(key=lambda i: -scores[i])
        else:
            keep_indices = [int(i) for i in np.argsort(-scores)]

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
