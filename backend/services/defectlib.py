"""
คลัง defect: เก็บลายเซ็น OD จาก dataset เดิม แล้วเอาไปเติมบนถุงมือสีใหม่

Two halves. The first reads a project that is already labelled and, for every
box of a chosen class, works out the optical density the defect added to the
rubber it was sitting on -- not the pixels, the density. That is the part that
survives being moved to another colour.

The second measures the target line: how bright its gloves are, how bright the
backlight behind them is, how much of the grain is sensor noise and how much is
rubber, and how noise grows with brightness. Those numbers are what the
insertion needs in order to land a defect at the right depth and with the right
graininess on a glove it has never seen.

Both halves are measurement, not modelling. Nothing here is fitted or learned:
the density comes from a logarithm and the profile from medians and variances,
which is why they can be trusted on a line nobody has trained on yet.
"""

import json
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from config import REPO_ROOT
from services import projects
from services.atomicio import read_json, write_json_best_effort
from services.projects import ProjectError

DEFECT_LIBRARY_DIR = Path(REPO_ROOT) / 'data' / 'defect-library'

# คำในชื่อ label -> ชนิดทางฟิสิกส์
#
# The physics only cares about one distinction: does light come straight
# through, or is it passing through more or less rubber than usual. Names are
# how a dataset says which, and this is the mapping for the vocabulary this
# factory uses. Anything unrecognised is treated as a stain, which is the
# conservative choice -- it changes density rather than punching a hole.
CLASS_KEYWORDS = (
    ('hole', 'hole'),
    ('pinhole', 'hole'),
    ('tear', 'tear'),
    ('split', 'tear'),
    ('crack', 'tear'),
    ('cut', 'tear'),
    ('thin', 'thin'),
    ('strip', 'thin'),
    ('dip', 'thick'),
    ('roll', 'thick'),
    ('fold', 'thick'),
    ('thick', 'thick'),
    ('lump', 'thick'),
    ('spot', 'particle'),
    ('particle', 'particle'),
    ('bubble', 'particle'),
    ('dirt', 'stain'),
    ('stain', 'stain'),
    ('mark', 'stain'),
)

DEFAULT_CLASS = 'stain'

# ขนาด patch texture ที่เก็บไว้คืนผิวยาง
TEXTURE_PATCH = 48
TEXTURE_COUNT = 8


def classify(label_name):
    """เดาชนิดทางฟิสิกส์จากชื่อ label"""
    lowered = str(label_name or '').lower()
    for keyword, kind in CLASS_KEYWORDS:
        if keyword in lowered:
            return kind
    return DEFAULT_CLASS


def _slug(text):
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '-', str(text or '')).strip('-.')
    return (cleaned or 'item')[:60]


def library_dir(source_project):
    folder = DEFECT_LIBRARY_DIR / _slug(source_project)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ── ครึ่งแรก: เก็บลายเซ็นจาก dataset เดิม ──────────────────────────────────


def _ring_reference(gray, x1, y1, x2, y2):
    """ค่าความสว่างของยางปกติรอบๆ กล่อง ใช้เป็นฐานเทียบ OD"""
    height, width = gray.shape[:2]
    pad = max(6, int(round(0.35 * max(x2 - x1, y2 - y1))))
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(width, x2 + pad), min(height, y2 + pad)

    outer = gray[ry1:ry2, rx1:rx2]
    if outer.size == 0:
        return None

    keep = np.ones(outer.shape, bool)
    keep[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
    ring = outer[keep]
    if ring.size < 32:
        return None
    # Median, not mean: the ring often clips a second defect or the glove edge,
    # and one bright sliver would drag a mean far enough to invert the sign of
    # the density that comes out of it.
    return float(np.median(ring))


def harvest(source_project, labels=None, per_label=40, settings=None):
    """
    อ่านโปรเจกต์ที่ label ไว้แล้ว เก็บลายเซ็น OD ของทุกกล่องที่เลือก

    เก็บเป็น .npz หนึ่งไฟล์ต่อหนึ่ง defect พร้อม meta.json รวม
    """
    from services.imaging import imread

    projects.get_project(source_project)
    settings = settings or {}
    wanted = {str(name) for name in (labels or [])}

    folder = library_dir(source_project)
    for stale in folder.glob('*.npz'):
        stale.unlink(missing_ok=True)

    src_mm = float(settings.get('src_mm_per_px') or 0.0)
    src_psf = float(settings.get('src_psf_sigma') or 1.0)

    kept, per_count, skipped = [], {}, {'no_image': 0, 'flat': 0, 'tiny': 0}
    for entry in projects.list_images(source_project):
        if entry.get('augmented'):
            continue
        stored = projects.read_annotation(source_project, entry['filename'])
        regions = (stored or {}).get('regions') or []
        if not regions:
            continue
        if all(r.get('tag') not in wanted for r in regions) and wanted:
            continue

        bgr = imread(projects.images_dir(source_project) / entry['filename'])
        if bgr is None:
            skipped['no_image'] += 1
            continue
        import cv2
        gray = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr)
        gray = gray.astype(np.float32) + 1.0

        for region in regions:
            tag = str(region.get('tag') or '')
            if wanted and tag not in wanted:
                continue
            if per_count.get(tag, 0) >= per_label:
                continue

            x1 = int(max(0, round(region['x'])))
            y1 = int(max(0, round(region['y'])))
            x2 = int(min(gray.shape[1], round(region['x'] + region['width'])))
            y2 = int(min(gray.shape[0], round(region['y'] + region['height'])))
            if x2 - x1 < 6 or y2 - y1 < 6:
                skipped['tiny'] += 1
                continue

            reference = _ring_reference(gray, x1, y1, x2, y2)
            if reference is None or reference <= 1.0:
                skipped['flat'] += 1
                continue

            patch = gray[y1:y2, x1:x2]
            # ลายเซ็นคือส่วนต่างของ OD เทียบยางรอบๆ ไม่ใช่ค่า pixel
            delta_od = (np.log(reference) - np.log(np.maximum(patch, 1e-3))
                        ).astype(np.float32)

            spread = float(np.percentile(np.abs(delta_od), 98))
            if spread < 0.02:
                skipped['flat'] += 1
                continue
            mask = ((np.abs(delta_od) > spread * 0.35).astype(np.uint8) * 255)
            if mask.sum() == 0:
                skipped['flat'] += 1
                continue

            kind = settings.get('classes', {}).get(tag) or classify(tag)
            name = f'{_slug(tag)}-{len(kept):04d}.npz'
            np.savez_compressed(folder / name, delta_od=delta_od, mask=mask)
            kept.append({
                'file': name,
                'tag': tag,
                'defect_class': kind,
                'width': int(x2 - x1),
                'height': int(y2 - y1),
                'od_peak': round(spread, 4),
                'from_image': entry['filename'],
            })
            per_count[tag] = per_count.get(tag, 0) + 1

    meta = {
        'source_project': source_project,
        'harvested_at': datetime.now().isoformat(),
        'src_mm_per_px': src_mm,
        'src_psf_sigma': src_psf,
        'defects': kept,
        'skipped': skipped,
    }
    (folder / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return summary(source_project)


def load(source_project):
    """โหลด meta ของคลัง"""
    folder = library_dir(source_project)
    path = folder / 'meta.json'
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return None


def summary(source_project):
    """สรุปว่าคลังมี defect อะไรกี่ชิ้น"""
    meta = load(source_project) or {}
    per_tag = {}
    for item in meta.get('defects') or []:
        bucket = per_tag.setdefault(item['tag'], {'tag': item['tag'],
                                                  'defect_class': item['defect_class'],
                                                  'count': 0})
        bucket['count'] += 1
    return {
        'source_project': source_project,
        'harvested_at': meta.get('harvested_at'),
        'total': len(meta.get('defects') or []),
        'per_tag': sorted(per_tag.values(), key=lambda row: -row['count']),
        'skipped': meta.get('skipped') or {},
    }


def read_defect(source_project, item, label_ids):
    """อ่าน defect หนึ่งชิ้นจากคลัง ให้พร้อมส่งเข้า insert_defect"""
    meta = load(source_project) or {}
    folder = library_dir(source_project)
    with np.load(folder / item['file']) as data:
        delta_od = data['delta_od'].astype(np.float32)
        mask = data['mask'].astype(np.uint8)
    return {
        'delta_od': delta_od,
        'mask': mask,
        'label_id': int(label_ids.get(item['tag'], 0)),
        'defect_class': item['defect_class'],
        'src_mm_per_px': meta.get('src_mm_per_px') or 0.0,
        'src_psf_sigma': meta.get('src_psf_sigma') or 1.0,
        'tag': item['tag'],
    }


# ── ครึ่งหลัง: วัดสภาพแสงของไลน์ปลายทาง ────────────────────────────────────


def split_glove(gray):
    """
    แยกถุงมือออกจากพื้นหลัง โดยไม่เดาว่าฝั่งไหนสว่างกว่า

    Returns (glove_mask, glove_level, bg_level).

    Which side is the glove is decided by which one owns the border of the
    frame. A background fills the edges by definition -- it is what the glove
    is sitting in front of -- while a glove photographed to be inspected is
    somewhere in the middle. That holds whether it is a dark silhouette on a
    backlight or a bright one on a dark stage, and assuming either brightness
    outright is what put every defect on the background.
    """
    import cv2

    eight = np.clip(gray, 0, 255).astype(np.uint8)
    threshold, bright = cv2.threshold(eight, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = 255 - bright

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    def border_share(mask):
        edges = np.concatenate([mask[0, :], mask[-1, :],
                                mask[:, 0], mask[:, -1]])
        return float((edges > 0).mean())

    # The one that owns less of the border is the object in front.
    glove_mask = bright if border_share(bright) < border_share(dark) else dark
    other = dark if glove_mask is bright else bright

    if (glove_mask > 0).sum() < 200 or (other > 0).sum() < 200:
        return None, None, None

    return (glove_mask,
            float(np.median(gray[glove_mask > 0])),
            float(np.median(gray[other > 0])))


def estimate_profile(grays, settings=None):
    """
    วัดค่าออปติกของถุงมือสีเป้าหมายจากภาพดีที่ import เข้ามา

    ทุกค่าได้จากการวัด ไม่ได้ฝึกมา จึงใช้กับไลน์ที่ยังไม่เคยเทรนได้
    """
    import cv2

    settings = settings or {}
    stacked = [np.asarray(g, np.float32) for g in grays if g is not None]
    if not stacked:
        raise ProjectError('No readable images to measure the line from')

    glove_vals, bg_vals, noise_vals, shot_ratios, texture = [], [], [], [], []

    for gray in stacked[:24]:
        eight = np.clip(gray, 0, 255).astype(np.uint8)
        # ฝั่งไหนคือถุงมือ ตัดสินจากใครครองขอบภาพ ไม่ใช่จากความสว่าง
        glove_mask, glove_level, bg_level = split_glove(gray)
        if glove_mask is None:
            continue
        glove_vals.append(glove_level)
        bg_vals.append(bg_level)

        # noise = ส่วนที่เหลือหลังหักความสว่างพื้นถิ่นออก วัดบนยางเท่านั้น
        smooth = cv2.medianBlur(eight, 5).astype(np.float32)
        residual = gray - smooth
        inside = residual[glove_mask > 0]
        if inside.size > 500:
            # MAD, because the residual also holds every bit of real grain and
            # every edge; a standard deviation would be reporting those.
            noise_vals.append(float(np.median(np.abs(inside)) * 1.4826))

        # k_shot: var = mean * k วัดจากบล็อกเล็กๆ บนยาง
        #
        # Measured over the rubber only. Taken over the whole frame it is the
        # background that dominates, and a flat dark background reports a
        # variance of nothing -- which came out as noise 0 and left the
        # signal-to-noise gate unable to reject anything at all.
        step = 16
        height = gray.shape[0] // step * step
        width = gray.shape[1] // step * step
        cropped = gray[:height, :width]
        mask_crop = (glove_mask[:height, :width] > 0).astype(np.float32)
        blocks = cropped.reshape(height // step, step,
                                 width // step, step)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, step * step)
        mask_blocks = mask_crop.reshape(height // step, step,
                                        width // step, step)
        mask_blocks = mask_blocks.transpose(0, 2, 1, 3).reshape(-1, step * step)
        on_rubber = mask_blocks.mean(axis=1) > 0.95
        means = blocks.mean(axis=1)
        variances = blocks.var(axis=1)
        # เอาเฉพาะบล็อกเรียบที่อยู่บนยางทั้งบล็อก
        flat = variances < np.percentile(variances, 40)
        usable = flat & (means > 5) & on_rubber
        if usable.sum() > 20:
            shot_ratios.extend((variances[usable] / means[usable]).tolist())
            flat_blocks = blocks[usable]
            for row in flat_blocks[:TEXTURE_COUNT]:
                patch = row.reshape(step, step)
                texture.append((patch - patch.mean()).astype(np.float32))

    if not glove_vals:
        raise ProjectError(
            'Could not separate glove from background in these images. They '
            'should be backlit frames of a single glove, as the camera sees '
            'them on the line.')

    profile = {
        'glove_level': float(np.median(glove_vals)),
        'bg_level': float(np.median(bg_vals)),
        # A floor on both: a measurement of zero is not a quiet camera, it is
        # a measurement that failed, and passing it on disables the gate that
        # throws out defects too faint to see.
        'noise_sigma': max(float(np.median(noise_vals)) if noise_vals else 1.5,
                           0.5),
        'k_shot': max(float(np.median(shot_ratios)) if shot_ratios else 0.02,
                      1e-3),
        # PSF cannot be read off a picture of an unknown object, so it is a
        # setting with a sane default rather than a number invented here. One
        # pixel is right for a lens focused on a sensor of this size; the
        # consequence of being wrong is a defect slightly too sharp or soft,
        # not a wrong grey.
        'psf_sigma': float(settings.get('psf_sigma') or 1.0),
        'mm_per_px': float(settings.get('mm_per_px') or 0.0),
        'texture_bank': texture[:TEXTURE_COUNT],
        'measured_from': len(stacked),
    }
    return profile


def glove_area(gray):
    """
    หาพื้นที่ยางที่วาง defect ได้ โดยไม่เดาว่าถุงมือสว่างหรือมืด

    A defect on the background is not a defect, it is a mark on the machine
    behind the glove -- and training on it teaches the detector to look there.
    Which is what happened while this assumed the glove was always the darker
    side.
    """
    import cv2

    mask, _, _ = split_glove(gray)
    if mask is None:
        return np.zeros(gray.shape[:2], np.uint8)

    # เอาเฉพาะชิ้นใหญ่สุด ไม่ให้เศษเล็กๆ กลายเป็นที่วาง
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = ((labels == largest).astype(np.uint8) * 255)
    return mask


def _positions(mask, count, size, rng):
    """สุ่มตำแหน่งวางบนยาง ห่างขอบพอที่ defect จะอยู่บนยางทั้งชิ้น"""
    import cv2

    margin = max(8, int(size * 0.7))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (margin * 2 + 1, margin * 2 + 1))
    inner = cv2.erode(mask, kernel)
    ys, xs = np.nonzero(inner)
    if len(xs) == 0:
        # ถุงมือเล็กกว่า defect ยอมวางกลางพื้นที่ยางแทนที่จะไม่วางเลย
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
    picks = rng.choice(len(xs), size=min(count, len(xs)), replace=False)
    return [(int(xs[i]), int(ys[i])) for i in picks]


# ── งานเติมเป็นชุด ─────────────────────────────────────────────────────────


_locks = {}
_locks_guard = threading.Lock()


def _lock(name):
    with _locks_guard:
        return _locks.setdefault(name, threading.Lock())


def status_path(name):
    return projects.training_dir(name) / 'defect_synth_status.json'


def get_status(name):
    projects.get_project(name)
    return read_json(status_path(name))


def _write_status(name, **fields):
    status = read_json(status_path(name)) or {}
    status.update(fields)
    write_json_best_effort(status_path(name), status)
    return status


def start(target_project, source_project, tags, per_image=1, settings=None):
    """เริ่มงานเติม defect ลงภาพถุงมือดีของโปรเจกต์เป้าหมาย (เบื้องหลัง)"""
    projects.get_project(target_project)
    meta = load(source_project)
    if not meta:
        raise ProjectError(
            f'No defect library for "{source_project}" yet. Collect one from a '
            'project that is already labelled first.')

    chosen = [d for d in meta['defects'] if not tags or d['tag'] in set(tags)]
    if not chosen:
        raise ProjectError('None of the chosen labels are in that library')

    lock = _lock(target_project)
    if not lock.acquire(blocking=False):
        raise ProjectError('A defect run is already going for this project')

    _write_status(target_project, status='running', done=0, made=0, refused=0,
                  total=0, started_at=datetime.now().isoformat(),
                  source_project=source_project, tags=list(tags or []),
                  message='Measuring the line')

    thread = threading.Thread(
        target=_run,
        args=(target_project, source_project, chosen, per_image, settings or {}, lock),
        daemon=True)
    thread.start()
    return get_status(target_project)


def _run(target, source, chosen, per_image, settings, lock):
    try:
        _synthesise(target, source, chosen, per_image, settings)
    except Exception as exc:  # noqa: BLE001 - reported, never raised into a thread
        _write_status(target, status='failed',
                      message=f'{type(exc).__name__}: {str(exc)[:300]}',
                      finished_at=datetime.now().isoformat())
    finally:
        try:
            lock.release()
        except RuntimeError:
            pass


def _synthesise(target, source, chosen, per_image, settings):
    import cv2

    from services import defectsynth
    from services.imaging import imread

    # ภาพดีคือภาพที่ยังไม่มีกล่อง -- ของเสียมี label อยู่แล้ว
    entries = [e for e in projects.list_images(target)
               if not e.get('augmented') and not e.get('annotated')]
    batch_filter = settings.get('batch')
    if batch_filter is not None:
        entries = [e for e in entries
                   if (projects.read_annotation(target, e['filename']) or {})
                   .get('batch') == int(batch_filter)]
    if not entries:
        raise ProjectError(
            'No un-annotated images to work on. Import the good gloves for this '
            'line first; anything already boxed is left alone.')

    limit = settings.get('limit')
    if limit:
        entries = entries[:int(limit)]

    images_dir = projects.images_dir(target)
    grays = []
    for entry in entries[:24]:
        bgr = imread(images_dir / entry['filename'])
        if bgr is None:
            continue
        grays.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                     if bgr.ndim == 3 else bgr.astype(np.float32))

    profile = estimate_profile(grays, settings)
    _write_status(target, total=len(entries),
                  profile={k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in profile.items() if k != 'texture_bank'},
                  message=f'Glove {profile["glove_level"]:.0f}, '
                          f'backlight {profile["bg_level"]:.0f}')

    # label ids ของโปรเจกต์เป้าหมาย: ชื่อที่ยังไม่มีจะถูกเพิ่มต่อท้าย
    existing = list(projects.class_names(target) or [])
    for item in chosen:
        if item['tag'] not in existing:
            existing.append(item['tag'])
    label_ids = {name: index for index, name in enumerate(existing)}

    rng = np.random.default_rng(1234)
    batch = projects.next_batch_number(target)
    imported_at = datetime.now().isoformat()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    done = made = refused = 0
    order = 0
    for entry in entries:
        if (read_json(status_path(target)) or {}).get('cancel_requested'):
            _write_status(target, status='cancelled', done=done, made=made,
                          refused=refused, finished_at=datetime.now().isoformat(),
                          message=f'Stopped after {made} image(s)')
            return

        done += 1
        bgr = imread(images_dir / entry['filename'])
        if bgr is None:
            continue
        gray = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr)

        area = glove_area(gray.astype(np.float32))
        working = gray.copy()
        regions = []

        for _ in range(max(1, int(per_image))):
            item = chosen[int(rng.integers(len(chosen)))]
            defect = read_defect(source, item, label_ids)
            size = max(defect['mask'].shape)
            spots = _positions(area, 1, size, rng)
            if not spots:
                continue

            result = defectsynth.insert_defect(working, defect, spots[0], profile)
            if not result['passed']:
                refused += 1
                continue

            working = result['image']
            _, cx, cy, w, h = result['bbox']
            height, width = working.shape[:2]
            regions.append({
                'tag': defect['tag'],
                'x': (cx - w / 2) * width,
                'y': (cy - h / 2) * height,
                'width': w * width,
                'height': h * height,
                'snr': result['snr'],
            })

        if not regions:
            continue

        order += 1
        # PNG, not JPEG. Two reasons, both fatal to the alternative. The
        # promise of this whole module is that every pixel outside the defect
        # is the good glove's own; JPEG changes all of them. And the SNR gate
        # accepts a defect at a measured contrast against measured noise --
        # quantising the image afterwards can take a defect that just cleared
        # the gate and remove it, leaving a box around nothing, which is the
        # worst thing this could put into a training set.
        new_name = f'{stamp}_syn_{order:05d}.png'
        cv2.imwrite(str(images_dir / new_name), working)
        projects.write_annotation(target, new_name, {
            'filename': new_name,
            'regions': regions,
            'annotated': True,
            'width': int(working.shape[1]),
            'height': int(working.shape[0]),
            'original_name': entry['filename'],
            'batch': batch,
            'imported_at': imported_at,
            # Not a prediction and not a person's drawing: a defect that was
            # put there on purpose, and whose box is therefore exactly right.
            # Saying so keeps it out of the review queue and lets a training
            # run be told how much of its data was made rather than seen.
            'synthetic': {
                'source_project': source,
                'from_image': entry['filename'],
                'glove_level': round(profile['glove_level'], 2),
                'at': datetime.now().isoformat(),
            },
        })
        made += 1

        if order % 10 == 0:
            _write_status(target, done=done, made=made, refused=refused,
                          message=f'{made} image(s) made, {refused} too faint')

    projects.rebuild_index(target)
    projects.refresh_stats(target)
    _write_status(target, status='finished', done=done, made=made,
                  refused=refused, batch=batch,
                  finished_at=datetime.now().isoformat(),
                  message=f'{made} image(s) made, {refused} refused as too '
                          f'faint to see, batch {batch}')


def cancel(name):
    projects.get_project(name)
    status = read_json(status_path(name)) or {}
    if status.get('status') != 'running':
        return {'cancelled': False, 'reason': 'nothing running'}
    _write_status(name, cancel_requested=True)
    return {'cancelled': True}


def wait_for_idle(name, timeout=600):
    """สำหรับเทสต์: รอจนงานจบ"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = read_json(status_path(name)) or {}
        if status.get('status') in ('finished', 'failed', 'cancelled'):
            return status
        time.sleep(0.2)
    return read_json(status_path(name)) or {}


def forget(source_project):
    """ลบคลังของโปรเจกต์ต้นทาง"""
    folder = DEFECT_LIBRARY_DIR / _slug(source_project)
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)
    return {'removed': folder.name}
