"""
Bringing in a dataset that was labelled somewhere else.

Six thousand pictures already boxed in Custom Vision are worth more than any
model exported from it: a model is a frozen answer, the pictures can be
trained again, augmented, corrected and extended. So they have to come in
whole, and the boxes have to land where they were drawn.

Two things here are easy to get wrong in ways nothing complains about. A YOLO
file says `8` and only label.txt knows that 8 is Good -- sorting those names
into alphabetical order would put a different word on every box in the set
while looking entirely correct. And a YOLO box is a normalised centre and
size, in a frame the annotation file never states, so the pixel dimensions
have to come from the image and no other guess will do.

Both are checked here by recomputing the answer from the source files rather
than by trusting the importer's own arithmetic.

    python backend/tests/test_dataset_import.py
"""
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp(prefix='vt_dsimport_'))
os.environ['PROJECTS_ROOT'] = str(TMP / 'projects')
os.environ['DATABASE_URL'] = 'sqlite:///' + (TMP / 't.db').as_posix()
os.environ['REQUIRE_AUTH'] = '0'
sys.path.insert(0, str(REPO / 'backend'))

import cv2                                              # noqa: E402
import numpy as np                                      # noqa: E402
from app import create_app                              # noqa: E402
from services import datasetimport, projects            # noqa: E402

app = create_app()
c = app.test_client()

fails = []


def check(label, cond, detail=''):
    print(f'  {"PASS " if cond else "FAIL "}{label}' + ('' if cond else f'  -> {detail}'))
    if not cond:
        fails.append(label)


# Deliberately not in alphabetical order, and with the interesting class late
# in the list: a sorted read would put the wrong name on every box, and only
# an out-of-order file catches that.
CLASS_NAMES = ['DirtM', 'DirtS', 'DoubleDip', 'WhiteSpot', 'CuffTear',
               'TearLine', 'Split', 'DirtL', 'Good']
WIDTH, HEIGHT = 1600, 1199

# One box per picture, as a normalised centre and size.
SOURCE = {
    'aaa': (8, 0.388942, 0.473257, 0.660577, 0.601282),   # Good
    'bbb': (4, 0.614423, 0.788001, 0.146154, 0.082051),   # CuffTear
    'ccc': (0, 0.266026, 0.579962, 0.460572, 0.607792),   # DirtM
}


def wanted(entry):
    """The pixel box the file describes, computed here and not by the code."""
    index, cx, cy, w, h = entry
    return (CLASS_NAMES[index],
            (cx - w / 2) * WIDTH, (cy - h / 2) * HEIGHT,
            w * WIDTH, h * HEIGHT)


def write_photo(path):
    frame = np.full((HEIGHT, WIDTH, 3), 40, np.uint8)
    cv2.rectangle(frame, (200, 200), (900, 800), (210, 215, 225), -1)
    # imencode then write the bytes: the folder name below is not ASCII, and
    # cv2.imwrite cannot open such a path on Windows.
    path.write_bytes(cv2.imencode('.jpg', frame)[1].tobytes())


# A folder name with characters outside ASCII, because the real export lives
# under one and OpenCV silently fails to read those paths on Windows.
export = TMP / 'เอกสาร' / 'export'
(export / 'images').mkdir(parents=True)
(export / 'labels').mkdir(parents=True)
(export / 'label.txt').write_text('\n'.join(CLASS_NAMES) + '\n', encoding='utf-8')
for stem, entry in SOURCE.items():
    write_photo(export / 'images' / f'{stem}.jpg')
    index, cx, cy, w, h = entry
    # No trailing newline, which is how the real export writes them.
    (export / 'labels' / f'{stem}.txt').write_text(
        f'{index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}', encoding='utf-8')

print('== looking before copying ==')
r = c.post('/api/projects', json={'name': 'gloves'})
check('the project is created', r.status_code in (200, 201), r.status_code)

r = c.post('/api/projects/gloves/dataset-import/preview',
           json={'folder': str(export)})
found = r.get_json() or {}
check('the folder is read', r.status_code == 200, found)
check('recognised as YOLO', found.get('format') == 'yolo', found.get('format'))
check('every picture counted', found.get('images') == 3, found.get('images'))
check('the class names came from label.txt',
      found.get('classes_from') == 'label.txt', found.get('classes_from'))
check('and in the order the file gives them, not sorted',
      found.get('classes') == CLASS_NAMES, found.get('classes'))
check('nothing to warn about', found.get('warnings') == [], found.get('warnings'))

print('\n== a folder with no annotations in it ==')
bare = TMP / 'bare'
bare.mkdir()
write_photo(bare / 'x.jpg')
r = c.post('/api/projects/gloves/dataset-import/preview', json={'folder': str(bare)})
check('is refused with what was expected instead', r.status_code == 400
      and 'YOLO' in (r.get_json() or {}).get('message', ''),
      (r.status_code, (r.get_json() or {}).get('message', '')[:120]))

r = c.post('/api/projects/gloves/dataset-import/preview', json={'folder': ''})
check('and so is no folder at all', r.status_code == 400, r.status_code)

print('\n== importing it ==')
r = c.post('/api/projects/gloves/dataset-import', json={'folder': str(export)})
check('the job starts', r.status_code == 200, r.get_json())

with app.app_context():
    status = datasetimport.wait_for_idle('gloves', timeout=120)
check('it finishes', status.get('status') == 'finished', status)
check('every picture came in', status.get('imported') == 3, status)
check('with its box', status.get('boxes') == 3, status)
check('and they are stamped as one batch', status.get('batch') == 1, status)

print('\n== the boxes are where the files said ==')
with app.app_context():
    entries = projects.list_images('gloves')
    stored = {}
    for entry in entries:
        record = projects.read_annotation('gloves', entry['filename'])
        stored[Path(record['original_name']).stem] = record

check('one annotation per picture', len(stored) == 3, sorted(stored))
for stem, entry in SOURCE.items():
    record = stored.get(stem) or {}
    regions = record.get('regions') or []
    tag, x, y, w, h = wanted(entry)
    if len(regions) != 1:
        check(f'{stem} has one box', False, regions)
        continue
    box = regions[0]
    check(f'{stem} is a {tag}', box['tag'] == tag, box['tag'])
    check(f'{stem} is where it was drawn',
          all(abs(a - b) < 0.01 for a, b in
              zip((box['x'], box['y'], box['width'], box['height']), (x, y, w, h))),
          {k: round(box[k], 1) for k in ('x', 'y', 'width', 'height')})
    check(f'{stem} knows the size it was measured against',
          (record.get('width'), record.get('height')) == (WIDTH, HEIGHT),
          (record.get('width'), record.get('height')))

print('\n== an imported box is somebody else\'s work, not a prediction ==')
sample = next(iter(stored.values()))
check('it is not marked as auto-labelled', not sample.get('auto_labelled'), sample.get('auto_labelled'))
check('but it does record where it came from',
      (sample.get('imported_from') or {}).get('format') == 'yolo',
      sample.get('imported_from'))
body = c.get('/api/projects/gloves/review/queue').get_json() or {}
check('so it does not fill the review queue', body.get('waiting') == 0,
      body.get('waiting'))

print('\n== COCO ==')
coco = TMP / 'coco'
(coco / 'images').mkdir(parents=True)
for stem in ('p', 'q'):
    write_photo(coco / 'images' / f'{stem}.jpg')
(coco / 'notes.json').write_text(json.dumps({
    'images': [{'id': 1, 'file_name': 'p.jpg'}, {'id': 2, 'file_name': 'q.jpg'}],
    'categories': [{'id': 5, 'name': 'CuffTear'}, {'id': 9, 'name': 'Hole'}],
    'annotations': [
        {'image_id': 1, 'category_id': 5, 'bbox': [100, 120, 240, 180]},
        {'image_id': 2, 'category_id': 9, 'bbox': [400, 300, 60, 90]},
    ],
}), encoding='utf-8')

r = c.post('/api/projects', json={'name': 'cocoproj'})
found = (c.post('/api/projects/cocoproj/dataset-import/preview',
                json={'folder': str(coco)}).get_json() or {})
check('recognised as COCO', found.get('format') == 'coco', found.get('format'))
check('with its categories', found.get('classes') == ['CuffTear', 'Hole'],
      found.get('classes'))

c.post('/api/projects/cocoproj/dataset-import', json={'folder': str(coco)})
with app.app_context():
    status = datasetimport.wait_for_idle('cocoproj', timeout=120)
    records = [projects.read_annotation('cocoproj', e['filename'])
               for e in projects.list_images('cocoproj')]
check('both pictures came in', status.get('imported') == 2, status)
found_boxes = {r['original_name']: (r['regions'][0]['tag'], r['regions'][0]['x'],
                                    r['regions'][0]['width'])
               for r in records if r.get('regions')}
check('the boxes are pixels already, and stay that way',
      found_boxes.get('p.jpg') == ('CuffTear', 100.0, 240.0), found_boxes)

print('\n== Pascal VOC ==')
voc = TMP / 'voc'
voc.mkdir()
write_photo(voc / 'one.jpg')
(voc / 'one.xml').write_text(
    '<annotation><object><name>Split</name><bndbox>'
    '<xmin>50</xmin><ymin>60</ymin><xmax>250</xmax><ymax>360</ymax>'
    '</bndbox></object></annotation>', encoding='utf-8')

r = c.post('/api/projects', json={'name': 'vocproj'})
found = (c.post('/api/projects/vocproj/dataset-import/preview',
                json={'folder': str(voc)}).get_json() or {})
check('recognised as VOC', found.get('format') == 'voc', found.get('format'))

c.post('/api/projects/vocproj/dataset-import', json={'folder': str(voc)})
with app.app_context():
    status = datasetimport.wait_for_idle('vocproj', timeout=120)
    records = [projects.read_annotation('vocproj', e['filename'])
               for e in projects.list_images('vocproj')]
check('the picture came in', status.get('imported') == 1, status)
region = (records[0].get('regions') or [{}])[0] if records else {}
check('corners became a position and a size',
      (region.get('tag'), region.get('x'), region.get('y'),
       region.get('width'), region.get('height')) == ('Split', 50.0, 60.0, 200.0, 300.0),
      region)

print('\n== the same export, zipped, through the button that was already there ==')
# The Import dataset button used to take only this application's own export
# and refuse everything else, which is the wrong way round: the datasets worth
# importing are the ones from somewhere else. A user zipped a Custom Vision
# export, pressed it, and got a 400 saying it had to be a .zip produced by
# Export -- which it could never be.
import zipfile                                          # noqa: E402

bundle = TMP / 'export.zip'
with zipfile.ZipFile(bundle, 'w') as archive:
    for path in export.rglob('*'):
        if path.is_file():
            # Nested under one folder, the way right-clicking a folder zips it.
            archive.write(path, Path('export') / path.relative_to(export))

r = c.post('/api/projects', json={'name': 'zipped'})
r = c.post('/api/projects/zipped/import-dataset',
           data={'file': (bundle.open('rb'), 'export.zip')},
           content_type='multipart/form-data')
body = r.get_json() or {}
check('a zip from another tool is accepted', r.status_code == 200,
      (r.status_code, body.get('message')))
check('and read as what it is', (body.get('job') or {}).get('format') == 'yolo',
      body.get('job'))

with app.app_context():
    status = datasetimport.wait_for_idle('zipped', timeout=120)
    zipped = [projects.read_annotation('zipped', e['filename'])
              for e in projects.list_images('zipped')]
check('every picture came in', status.get('imported') == 3, status)
tags = sorted({r['tag'] for rec in zipped for r in (rec.get('regions') or [])})
check('with the names its label.txt gave them',
      tags == ['CuffTear', 'DirtM', 'Good'], tags)

print('\n== a zip that is not a zip ==')
r = c.post('/api/projects/zipped/import-dataset',
           data={'file': (io.BytesIO(b'not a zip'), 'export.zip')},
           content_type='multipart/form-data')
check('is refused rather than crashing', r.status_code == 400,
      (r.status_code, (r.get_json() or {}).get('message', '')[:80]))

r = c.post('/api/projects/zipped/import-dataset',
           data={'file': (io.BytesIO(b'x'), 'export.rar')},
           content_type='multipart/form-data')
message = (r.get_json() or {}).get('message', '')
check('and a file that is not a zip says what to do instead',
      r.status_code == 400 and 'path' in message, message[:120])

print('\n== a zip cannot write outside where it is unpacked ==')
nasty = TMP / 'nasty.zip'
with zipfile.ZipFile(nasty, 'w') as archive:
    archive.writestr('../../escaped.txt', 'no')
    archive.writestr('images/ok.jpg', b'not an image but a plain member')
with app.app_context():
    folder = datasetimport.unpack(nasty, 'zipped')
check('the climbing member is dropped',
      not (TMP / 'escaped.txt').exists()
      and not (Path(folder).parent.parent / 'escaped.txt').exists(),
      sorted(p.name for p in Path(folder).rglob('*')))

print('\n== a second import is a second batch ==')
c.post('/api/projects/gloves/dataset-import', json={'folder': str(export)})
with app.app_context():
    status = datasetimport.wait_for_idle('gloves', timeout=120)
check('numbered so the newer set can be told from the older',
      status.get('batch') == 2, status.get('batch'))

print('\n' + ('DATASET IMPORT OK' if not fails else f'{len(fails)} FAILED: {fails}'))
if fails:
    print(f'(kept for inspection: {TMP})')
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
