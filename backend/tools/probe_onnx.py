"""
Find the conventions a detector ONNX was built with, by trying them.

An ONNX records the shape of what goes in and comes out, and nothing about
what the numbers mean. Four things are left unsaid, and every one of them
changes the answer:

    resize     padded to a square (letterbox), or squashed into one (stretch)
    channels   RGB, or BGR
    scale      pixels as 0..1, or as 0..255
    box order  x1 y1 x2 y2, or y1 x1 y2 x2

Get one wrong and nothing raises. The model returns confident nonsense --
usually a single box over the whole frame, which is what sent this tool into
being: a real glove-defect model answered `class_10  98%  6, 0, 1595, 1199` on
a 1600x1200 photo, which is the entire picture.

So every combination is run over real images and scored on what a working
detector does and a broken one does not:

  * its boxes cover part of the frame, not all of it
  * its boxes move between pictures
  * it does not answer the same class every time

Usage:

    python backend/tools/probe_onnx.py model.onnx folder-of-images
    python backend/tools/probe_onnx.py model.onnx folder --threshold 0.3

Ten images is plenty. Pictures whose answers differ -- some good, some
defective -- tell the most, because a configuration that cannot tell them
apart is not working whatever its boxes look like.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESIZES = ('letterbox', 'stretch')
CHANNELS = ('rgb', 'bgr')
SCALES = ('unit', 'raw')
ORDERS = ('xyxy', 'yxyx')

SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
# A box this much of the frame is the model saying "everything", which is what
# a wrongly-fed detector says.
WHOLE_FRAME = 0.85


def load_images(folder, limit):
    import cv2
    files = sorted(p for p in Path(folder).iterdir()
                   if p.suffix.lower() in SUFFIXES)[:limit]
    images = []
    for path in files:
        frame = cv2.imread(str(path))
        if frame is not None:
            images.append((path.name, frame))
    return images


def area_fraction(box, frame):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1) / float(max(width * height, 1))


def try_one(path, images, threshold, resize, channels, scale, order):
    from services.onnxrunner import OnnxDetector

    detector = OnnxDetector(path, display_name=Path(path).name, resize=resize,
                            channels=channels, scale=scale, box_order=order)

    per_image, areas, classes, scores = [], [], set(), []
    for _, frame in images:
        found = detector.predict(frame, threshold=threshold)
        per_image.append(len(found))
        for item in found:
            areas.append(area_fraction(item['box'], frame))
            classes.add(item['class_id'])
            scores.append(item['score'])

    total = sum(per_image)
    whole = sum(1 for a in areas if a >= WHOLE_FRAME)
    return {
        'layout': detector.layout,
        'images_with_something': sum(1 for n in per_image if n),
        'detections': total,
        'whole_frame': whole,
        'median_area': sorted(areas)[len(areas) // 2] if areas else 0.0,
        'classes': len(classes),
        'mean_score': sum(scores) / len(scores) if scores else 0.0,
    }


def write_previews(path, images, threshold, out_dir, rows):
    """
    Draw each surviving configuration's boxes onto the pictures.

    The numbers settle the channels and the pixel range: feed a model the
    wrong ones and it finds nothing at all. They do not settle the fitting or
    the box order -- both produce boxes that look reasonable in a table, and
    only a picture says which sits on the object. So the pictures get written
    and somebody looks at them.
    """
    import cv2
    from services.onnxrunner import OnnxDetector

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for row in rows:
        resize, channels, scale, order = row['label'].split()
        detector = OnnxDetector(path, display_name=Path(path).name,
                                resize=resize, channels=channels, scale=scale,
                                box_order=order)
        tag = f'{resize}-{channels}-{scale}-{order}'
        for name, frame in images:
            canvas = frame.copy()
            for item in detector.predict(frame, threshold=threshold):
                x1, y1, x2, y2 = item['box']
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 90), 2)
                cv2.putText(canvas, f'{item["class_id"]} {item["score"]:.2f}',
                            (x1 + 4, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (60, 220, 90), 2, cv2.LINE_AA)
            cv2.putText(canvas, tag, (8, canvas.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 2,
                        cv2.LINE_AA)
            target = out_dir / f'{tag}__{Path(name).stem}.jpg'
            cv2.imwrite(str(target), canvas)
            written.append(target)
    return written


def verdict(row, image_count):
    """What this configuration looks like, in three words."""
    if not row['detections']:
        return 'finds nothing'
    if row['whole_frame'] == row['detections']:
        return 'every box is the whole frame'
    if row['whole_frame'] > row['detections'] / 2:
        return 'mostly whole-frame boxes'
    if row['classes'] == 1 and image_count > 3:
        return 'one class for everything'
    return 'PLAUSIBLE -- boxes vary'


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('images', help='a folder of photos the model should work on')
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--out', default=None,
                        help='where to write the annotated pictures')
    args = parser.parse_args(argv[1:])

    model = Path(args.model)
    if not model.is_file():
        print(f'No such model: {model}')
        return 2

    images = load_images(args.images, args.limit)
    if not images:
        print(f'No readable images in {args.images}')
        return 2
    print(f'{model.name} over {len(images)} image(s), threshold {args.threshold}\n')

    rows = []
    for resize in RESIZES:
        for channels in CHANNELS:
            for scale in SCALES:
                for order in ORDERS:
                    label = f'{resize:9} {channels} {scale:4} {order}'
                    try:
                        row = try_one(model, images, args.threshold,
                                      resize, channels, scale, order)
                    except Exception as exc:  # noqa: BLE001
                        print(f'  {label}  failed: {type(exc).__name__}: '
                              f'{str(exc)[:90]}')
                        continue
                    row['label'] = label
                    rows.append(row)
                    print(f'  {label}  {row["detections"]:4} det  '
                          f'{row["images_with_something"]}/{len(images)} imgs  '
                          f'area {row["median_area"] * 100:5.1f}%  '
                          f'{row["classes"]} class(es)  '
                          f'conf {row["mean_score"]:.2f}   '
                          f'{verdict(row, len(images))}')

    plausible = [r for r in rows if verdict(r, len(images)).startswith('PLAUSIBLE')]
    print()
    if not plausible:
        print('Nothing here looks like working detection. Either this model '
              'wants preprocessing that is not in the list -- a mean and '
              'standard deviation, a fixed crop -- or these images are not '
              'what it was trained on. Send this whole output.')
        return 1

    print('Worth looking at, best first:')
    # Smaller boxes and more classes are the marks of a detector that is
    # actually discriminating rather than shrugging at the whole picture.
    ranked = sorted(plausible, key=lambda r: (r['median_area'], -r['classes']))
    for row in ranked:
        print(f'  {row["label"]}   median box {row["median_area"] * 100:.1f}% '
              f'of the frame, {row["classes"]} class(es), '
              f'{row["detections"]} detection(s)')
    print(f'\nThe layout was read as: {rows[0]["layout"]}')

    # The numbers cannot separate the fitting or the box order -- both give
    # boxes that read fine in a table and only a picture says which one sits
    # on the object.
    out_dir = Path(args.out) if args.out else model.parent / 'probe-previews'
    written = write_previews(model, images[:3], args.threshold, out_dir,
                             ranked[:6])
    print(f'\nWrote {len(written)} annotated picture(s) to {out_dir}')
    print('Open them. The one whose boxes sit on the object is the answer; '
          'a whole-frame box or a box mirrored across the diagonal is not.')
    print('Send the table and the name of the picture that looks right.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
