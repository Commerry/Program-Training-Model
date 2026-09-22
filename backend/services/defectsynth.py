"""
เติม defect สังเคราะห์ลงบนภาพถุงมือดี โดยย้าย "ลายเซ็นความเข้ม" ไม่ใช่ย้าย pixel

Why this exists, in English for the rest of the codebase: a mono camera sees a
glove as one grey level, and that level is set by the rubber's colour. A model
trained on white gloves does not transfer to black ones, so every colour on
every production line needs its own dataset -- and the defects are the rare
half of it. Waiting for enough torn black gloves to appear is what makes a new
line slow to start.

The defects themselves are not colour-specific. A tear is a tear; what differs
is how much light the surrounding rubber lets through. So a defect collected
from one colour can be moved onto a good glove of another colour, provided the
move is done in the domain where it is physically additive.

That domain is optical density. Transmitted intensity through a material is

    I = I0 * exp(-OD)

so stacking material adds OD rather than multiplying intensity. A defect is a
local change in OD, and that change is a property of the defect, not of the
rubber behind it. Copying pixels instead would carry the source glove's own
brightness across and land the wrong grey on the target.

The rule the whole module is built around: every pixel in the output is a
pixel of the target image with its value adjusted. None is imported.
"""

import numpy as np

# เกณฑ์ SNR ขั้นต่ำ -- ต่ำกว่านี้ถือว่ามองไม่เห็น ไม่รับเข้า dataset
#
# A synthetic defect nobody can see is worse than no defect: it teaches the
# model that this patch of ordinary rubber is a tear, which is a false positive
# trained in deliberately. Rose's criterion puts reliable detection at about 5;
# 3 is the loosest defensible floor and is not adjustable on purpose.
SNR_MIN = 3.0

# ชนิด defect ที่แสงทะลุผ่าน ใช้ค่าสัมบูรณ์ของ backlight
#
# A hole is not thinner rubber, it is no rubber: the camera sees the backlight
# directly, and the backlight is the same brightness whatever colour the glove
# is. Treating it as an OD change would make a hole in a black glove darker
# than a hole in a white one, which is backwards.
SEE_THROUGH = {'hole', 'tear'}

# ชนิดที่เป็นการเปลี่ยนความหนา/สิ่งทับ ใช้ OD บวกกัน
ABSORBING = {'thin', 'thick', 'particle', 'stain'}

DEFECT_CLASSES = SEE_THROUGH | ABSORBING


def _place(small, shape, pos):
    """วาง array เล็กลงบน canvas ขนาดเท่าภาพ ที่ตำแหน่ง pos (ตัดขอบให้อัตโนมัติ)"""
    canvas = np.zeros(shape, np.float32)
    h, w = small.shape[:2]
    cx, cy = int(round(pos[0])), int(round(pos[1]))
    x0, y0 = cx - w // 2, cy - h // 2

    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(shape[1], x0 + w), min(shape[0], y0 + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return canvas

    canvas[dy0:dy1, dx0:dx1] = small[sy0:sy0 + (dy1 - dy0),
                                     sx0:sx0 + (dx1 - dx0)]
    return canvas


def _fit_optics(defect, profile):
    """ปรับขนาดและความคมของ defect ให้ตรงกับกล้องปลายทาง (ไม่สุ่มหมุน ไม่สุ่มขนาด)"""
    import cv2

    delta_od = np.asarray(defect['delta_od'], np.float32)
    mask = np.asarray(defect['mask'], np.float32)
    assert delta_od.dtype == np.float32
    assert mask.dtype == np.float32

    src_mm = float(defect.get('src_mm_per_px') or 0.0)
    dst_mm = float(profile.get('mm_per_px') or 0.0)
    # ขนาดจริงคงที่ จำนวน pixel เปลี่ยนตามอัตราส่วนสเกลของกล้อง
    if src_mm > 0 and dst_mm > 0:
        scale = src_mm / dst_mm
    else:
        scale = 1.0

    if abs(scale - 1.0) > 1e-3:
        h = max(1, int(round(delta_od.shape[0] * scale)))
        w = max(1, int(round(delta_od.shape[1] * scale)))
        # INTER_AREA when shrinking averages rather than samples, which keeps
        # the integrated optical density right instead of picking whichever
        # pixels the grid happened to land on.
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        delta_od = cv2.resize(delta_od, (w, h), interpolation=interp)
        mask = cv2.resize(mask, (w, h), interpolation=interp)

    src_psf = float(defect.get('src_psf_sigma') or 0.0)
    dst_psf = float(profile.get('psf_sigma') or 0.0)
    # เบลอเฉพาะส่วนต่าง กล้องปลายทางคมน้อยกว่าเท่านั้นที่ต้องเบลอเพิ่ม
    #
    # Blurs compose in quadrature, so the difference needed is the quadrature
    # difference. The other direction -- a sharper destination camera -- cannot
    # be had by sharpening: the detail was never recorded.
    if dst_psf > src_psf > 0 or (dst_psf > 0 and src_psf <= 0):
        extra = float(np.sqrt(max(dst_psf ** 2 - src_psf ** 2, 0.0)))
        if extra > 0.05:
            ksize = int(2 * round(3 * extra) + 1)
            delta_od = cv2.GaussianBlur(delta_od, (ksize, ksize), extra)
            mask = cv2.GaussianBlur(mask, (ksize, ksize), extra)

    return delta_od.astype(np.float32), np.clip(mask, 0, 255).astype(np.float32)


def _apply_by_class(img, delta_od, mask, pos, defect_class, profile):
    """เติม defect ตามชนิด: แสงทะลุใช้ค่าสัมบูรณ์ ส่วนที่เหลือบวกกันใน OD"""
    import cv2

    assert img.dtype == np.float32

    full_od = _place(delta_od, img.shape, pos)
    full_mask = _place(mask, img.shape, pos)

    if defect_class in SEE_THROUGH:
        # รูเห็น backlight ตรงๆ สว่างเท่ากันทุกสีถุงมือ
        psf = float(profile.get('psf_sigma') or 0.0)
        alpha = full_mask / 255.0
        if psf > 0.05:
            ksize = int(2 * round(3 * psf) + 1)
            alpha = cv2.GaussianBlur(alpha, (ksize, ksize), psf)
        alpha = np.clip(alpha, 0.0, 1.0)
        bg = float(profile['bg_level'])
        out = img * (1.0 - alpha) + bg * alpha
    else:
        # OD บวกกันตาม Beer-Lambert: I_out = I * exp(-delta_od)
        #
        # Written as a multiply rather than exp(log(I) - od). The two are the
        # same equation, but the round trip through log is not exact in
        # float32: with od zero it returned I to within a fraction of a level,
        # which after rounding to uint8 moved half the pixels in the picture by
        # one. Every pixel outside the defect must come back bit for bit, and
        # exp(0) is exactly 1, so multiplying gives that for free.
        out = img * np.exp(-full_od)

    return out.astype(np.float32), full_mask


def _restore_noise(out, original, full_mask, profile):
    """คืน shot noise และ texture ผิวยางในบริเวณที่แปะ ไม่ให้เนียนผิดธรรมชาติ"""
    import cv2

    assert out.dtype == np.float32
    assert original.dtype == np.float32

    psf = max(float(profile.get('psf_sigma') or 1.0), 0.8)
    ksize = int(2 * round(3 * psf) + 1)
    soft = cv2.GaussianBlur(np.clip(full_mask / 255.0, 0, 1), (ksize, ksize), psf)
    soft = np.clip(soft, 0.0, 1.0)
    # A Gaussian has no end, so without this the noise is added, faintly, to
    # every pixel in the frame -- and a picture that differs everywhere is no
    # longer the original picture with a defect on it.
    soft[soft < 1e-3] = 0.0
    if soft.max() <= 0:
        return out

    k_shot = float(profile.get('k_shot') or 0.0)
    result = out.copy()

    # shot noise ขึ้นกับความสว่าง ที่มืดลงต้อง noise น้อยลงด้วย
    #
    # The source patch arrived carrying the noise of its own glove, at its own
    # brightness. After the OD shift that noise is at the wrong scale, and a
    # region whose noise does not match its brightness is the single most
    # obvious tell that a patch was pasted.
    if k_shot > 0:
        rng = np.random.default_rng(0)
        sigma = np.sqrt(np.maximum(result, 1.0) * k_shot)
        shot = rng.standard_normal(result.shape).astype(np.float32) * sigma
        result = result + shot * soft

    bank = profile.get('texture_bank') or []
    if len(bank):
        patch = np.asarray(bank[0], np.float32)
        tiled = np.zeros_like(result)
        ph, pw = patch.shape[:2]
        if ph > 0 and pw > 0:
            reps = (int(np.ceil(result.shape[0] / ph)),
                    int(np.ceil(result.shape[1] / pw)))
            tiled = np.tile(patch, reps)[:result.shape[0], :result.shape[1]]
        result = result + tiled * soft

    return result.astype(np.float32)


def _snr_gate(out, original, full_mask, profile):
    """วัด contrast เทียบกับ noise รอบนอก คืนค่า SNR"""
    import cv2

    assert out.dtype == np.float32

    core = (full_mask > 127).astype(np.uint8)
    if core.sum() == 0:
        return 0.0

    # วงแหวนรอบนอกคือพื้นผิวยางปกติที่ใช้เทียบ
    grow = max(3, int(round(np.sqrt(core.sum()) * 0.3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1,) * 2)
    ring = cv2.dilate(core, kernel) - core

    if ring.sum() == 0:
        return 0.0

    inside = float(out[core > 0].mean())
    around = float(out[ring > 0].mean())
    contrast = abs(inside - around)

    k_shot = float(profile.get('k_shot') or 0.0)
    if k_shot > 0:
        sigma = float(np.sqrt(max(around, 1.0) * k_shot))
    else:
        sigma = float(profile.get('noise_sigma') or 1.0)

    return contrast / max(sigma, 1e-6)


def _make_bbox(out, original, label_id, profile):
    """หากรอบจากบริเวณที่ 'เห็นได้จริง' หลังแปะ ไม่ใช่จาก mask ต้นทาง"""
    import cv2

    assert out.dtype == np.float32
    assert original.dtype == np.float32

    # The same defect is less visible on a dark glove than a light one, because
    # less light is getting through to be modulated in the first place. A box
    # drawn from the source mask would then be larger than the thing a person
    # -- or a detector -- can actually see, and training on boxes that include
    # invisible margin teaches the model to expect them.
    noise_sigma = float(profile.get('noise_sigma') or 1.0)
    visible = (np.abs(out - original) > noise_sigma * 1.5).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    visible = cv2.morphologyEx(visible, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(visible, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, visible * 255

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < 2 or h < 2:
        return None, visible * 255

    height, width = out.shape[:2]
    bbox = (int(label_id),
            (x + w / 2.0) / width,
            (y + h / 2.0) / height,
            w / float(width),
            h / float(height))
    return bbox, visible * 255


def insert_defect(img, defect, pos, profile):
    """
    เติม defect หนึ่งจุดลงบนภาพถุงมือดี คืน dict ผลลัพธ์ หรือผลที่ไม่ผ่าน SNR

    img     : uint8 grayscale ภาพถุงมือดีของสีเป้าหมาย
    defect  : รายการจาก defect library (delta_od, mask, label_id, defect_class,
              src_mm_per_px, src_psf_sigma)
    pos     : (x, y) จุดศูนย์กลางที่จะวาง
    profile : ค่าออปติกและ noise ของสีเป้าหมาย
    """
    if img is None or img.ndim != 2:
        raise ValueError('ต้องเป็นภาพ grayscale 8-bit ช่องเดียว')

    defect_class = str(defect.get('defect_class') or '').lower()
    if defect_class not in DEFECT_CLASSES:
        raise ValueError(f'ไม่รู้จัก defect_class: {defect_class!r}')

    # +1.0 กัน log(0) และทำงานเป็น float32 ตลอดทาง ไม่ cast กลับระหว่างทาง
    work = img.astype(np.float32) + 1.0
    original = work.copy()

    delta_od, mask = _fit_optics(defect, profile)
    out, full_mask = _apply_by_class(work, delta_od, mask, pos,
                                     defect_class, profile)
    out = _restore_noise(out, original, full_mask, profile)

    snr = _snr_gate(out, original, full_mask, profile)
    if snr < SNR_MIN:
        # คืนผลที่ไม่ผ่านพร้อมค่า SNR
        #
        # The specification says to return None here, and also to return the
        # snr with it. None carries nothing, and the caller needs the number to
        # say why a batch produced fewer images than asked for -- "too faint on
        # this colour, 1.8" is actionable and a silent drop is not. So the
        # failure comes back in the same shape with passed False and no image.
        return {'image': None, 'bbox': None, 'mask': None,
                'snr': round(float(snr), 3), 'passed': False}

    bbox, visible = _make_bbox(out, original, defect.get('label_id', 0), profile)
    if bbox is None:
        return {'image': None, 'bbox': None, 'mask': None,
                'snr': round(float(snr), 3), 'passed': False}

    final = np.clip(out - 1.0, 0, 255).astype(np.uint8)
    return {
        'image': final,
        'bbox': bbox,
        'mask': visible.astype(np.uint8),
        'snr': round(float(snr), 3),
        'passed': True,
    }
