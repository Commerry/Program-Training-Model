"""
Colour-tone augmentation.

Generates extra training images from annotated originals by applying contrast,
edge and colour presets. Bounding boxes are unchanged because every preset is a
per-pixel or filter operation with no geometric component, so the original
annotation is copied verbatim.
"""

from datetime import datetime

import cv2
import numpy as np

from services import projects
from services.imaging import imread, imwrite
from services.projects import ProjectError

# Presets, grouped by what they emphasise. The B&W / edge presets are the most
# useful ones for reading embossed or low-contrast digits.
ALL_COLOR_TONES = [
    # colour
    'warm', 'cool', 'bright', 'dark', 'vivid', 'sepia', 'high_contrast', 'invert',
    # greyscale / contrast
    'gray', 'equalize', 'clahe', 'clahe_strong', 'sharpen', 'unsharp',
    'adaptive_thresh', 'tophat', 'blackhat', 'gamma_low', 'gamma_high', 'bilateral',
    # edge / contour emphasis
    'canny_overlay', 'laplacian', 'sobel', 'emboss', 'contour_inv', 'clahe_sharp',
]

# Windows still caps a full path at 260 characters unless long paths are
# enabled, and augmented names are built from the source name. Truncating the
# stem keeps generated names bounded no matter how deep the project lives.
MAX_STEM = 40

# An augmentation run happens inside the request, so it has to have a ceiling.
# Every source image is decoded, filtered, encoded and written once per tone
# per variant: 100 images with all 26 presets at 3 variants each is 7,800
# files, and nothing about the request tells the user that before it starts.
# Beyond this the request is refused with the arithmetic, rather than run
# until the client times out while the server keeps writing.
MAX_GENERATED_IMAGES = 5000


def apply_color_tone(image, tone, strength=1.0, seed=0):
    """Apply one preset to a BGR image and return the result."""
    img = image.copy()
    tone = (tone or '').lower().strip()
    strength = max(0.2, min(2.0, float(strength)))

    # deterministic random for slight variation per generated image
    rng = np.random.default_rng(seed)
    jitter = float(rng.uniform(0.92, 1.08))

    if tone == 'warm':
        b, g, r = cv2.split(img)
        r = cv2.convertScaleAbs(r, alpha=1.0 + 0.18 * strength * jitter, beta=6)
        b = cv2.convertScaleAbs(b, alpha=1.0 - 0.10 * strength * jitter, beta=0)
        img = cv2.merge([b, g, r])
    elif tone == 'cool':
        b, g, r = cv2.split(img)
        b = cv2.convertScaleAbs(b, alpha=1.0 + 0.18 * strength * jitter, beta=6)
        r = cv2.convertScaleAbs(r, alpha=1.0 - 0.10 * strength * jitter, beta=0)
        img = cv2.merge([b, g, r])
    elif tone == 'bright':
        img = cv2.convertScaleAbs(img, alpha=1.0 + 0.18 * strength * jitter, beta=int(14 * strength))
    elif tone == 'dark':
        img = cv2.convertScaleAbs(img, alpha=max(0.4, 1.0 - 0.20 * strength * jitter), beta=int(-14 * strength))
    elif tone == 'vivid':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= (1.0 + 0.45 * strength * jitter)
        hsv[..., 2] *= (1.0 + 0.08 * strength * jitter)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif tone == 'sepia':
        k = np.array([
            [0.131, 0.534, 0.272],
            [0.168, 0.686, 0.349],
            [0.189, 0.769, 0.393],
        ], dtype=np.float32)
        sep = cv2.transform(img, k)
        sep = np.clip(sep * (0.9 + 0.25 * strength * jitter), 0, 255).astype(np.uint8)
        img = sep
    elif tone == 'gray':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif tone == 'high_contrast':
        img = cv2.convertScaleAbs(img, alpha=1.0 + 0.28 * strength * jitter, beta=0)
    elif tone == 'invert':
        img = cv2.bitwise_not(img)
    elif tone == 'clahe':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip = float(np.clip(1.5 * strength * jitter, 1.0, 4.0))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    # ── New B&W / high-contrast filters ─────────────────────────
    elif tone == 'clahe_strong':
        # Stronger CLAHE (clip=4.0) — very effective for embossed numbers
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip = float(np.clip(4.0 * strength * jitter, 2.0, 8.0))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
        l2 = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    elif tone == 'equalize':
        # Histogram equalisation — spreads brightness levels across the range.
        #
        # equalizeHist works off the whole-frame histogram, which a small
        # object barely contributes to. Applied at full strength it stretched
        # the background across the range and squeezed the digit's narrow band
        # down with it, so the annotated region stopped separating from the
        # plate around it (measured separation fell from 115 to 7 out of 255).
        # Blending the result back over the original keeps the redistribution
        # while preserving the object's own contrast.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        weight = float(np.clip(0.55 * strength * jitter, 0.3, 0.8))
        blended = cv2.addWeighted(eq, weight, gray, 1.0 - weight, 0)
        img = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)
    elif tone == 'sharpen':
        # Sharpening kernel — enhances edges and digit boundaries.
        #
        # The coefficients must sum to 1, otherwise the filter changes overall
        # brightness as well as sharpness. The previous kernel summed to 0.5
        # and darkened every generated image by about half, which both wasted
        # the preset (it duplicated what 'dark' already does) and pushed the
        # digits toward the black end of the range.
        amount = float(np.clip(0.35 * strength * jitter, 0.1, 1.2))
        k = np.array([[-amount, -amount, -amount],
                      [-amount, 8 * amount + 1, -amount],
                      [-amount, -amount, -amount]], dtype=np.float32)
        img = cv2.filter2D(img, -1, k)
    elif tone == 'unsharp':
        # Unsharp mask — crisp fine details without harsh edges
        sigma = float(np.clip(1.5 * strength * jitter, 0.5, 3.0))
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        amount = float(np.clip(0.7 * strength * jitter, 0.3, 1.5))
        img = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
    elif tone == 'adaptive_thresh':
        # Adaptive thresholding → crisp B&W for embossed numbers on metal.
        #
        # A local threshold turns every grain of sensor noise into its own
        # black or white speck, so the untreated version filled the background
        # with salt-and-pepper that carried far more contrast than the digit
        # itself. Blur before the threshold, then open away whatever specks
        # survive, so what remains is the shape rather than the grain.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        # The block is the neighbourhood each pixel is compared against. Too
        # small and it mostly contains the digit itself, so the digit raises
        # its own threshold and stops standing out. Measured on a gauge frame,
        # separation between the digit and the plate beside it went from +151
        # at block 21 to +248 out of 255 at block 61.
        block = max(31, int(61 * strength) | 1)  # must be odd
        c_val = int(3 * strength * jitter)
        th = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, c_val)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        img = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    elif tone == 'tophat':
        # Morphological top-hat: extracts bright regions (raised digits)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ks = max(7, int(15 * strength) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        tophat = cv2.convertScaleAbs(tophat, alpha=2.5 * strength * jitter)
        img = cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR)
    elif tone == 'blackhat':
        # Morphological black-hat: extracts dark regions (sunken digits)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ks = max(7, int(15 * strength) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
        bhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        bhat = cv2.convertScaleAbs(bhat, alpha=2.5 * strength * jitter)
        # Black-hat responds to dark structures. On a bright object it finds
        # almost nothing, so on its own it produced a near-empty frame with the
        # annotation still pointing at it. Blending over the grayscale source
        # keeps the preset useful for sunken digits without emitting a blank
        # training image when the digits happen to be raised.
        blended = cv2.addWeighted(gray, 0.5, bhat, 0.5, 0)
        img = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)
    elif tone == 'gamma_low':
        # Darkens: pushes mid-tones down so a bright, washed-out plate regains
        # contrast.
        #
        # These two presets used to compute the same exponent — gamma_low took
        # 0.5/(s*j) = 0.5 and gamma_high took 1/(2*s*j) = 0.5 — so they emitted
        # pixel-identical images. Every augmentation run therefore produced two
        # copies of the same thing and one of the 26 presets was wasted.
        exponent = float(np.clip(1.8 * strength * jitter, 1.2, 3.0))
        table = np.array([(i / 255.0) ** exponent * 255 for i in range(256)],
                         dtype=np.uint8)
        img = cv2.LUT(img, table)
    elif tone == 'gamma_high':
        # Brightens: lifts shadows so digits in a dark frame become visible.
        exponent = float(np.clip(1.0 / (1.8 * strength * jitter), 0.33, 0.83))
        table = np.array([(i / 255.0) ** exponent * 255 for i in range(256)],
                         dtype=np.uint8)
        img = cv2.LUT(img, table)
    elif tone == 'bilateral':
        # Bilateral filter — smooth noise while keeping edges sharp
        d = int(np.clip(9 * strength, 5, 15))
        sc = float(np.clip(60 * strength * jitter, 30, 120))
        img = cv2.bilateralFilter(img, d, sc, sc)
    # ── Strong contour / edge-emphasis filters ──────────────────
    elif tone == 'canny_overlay':
        # Canny edges (white lines) blended onto grayscale — sees digit outlines clearly
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lo = int(np.clip(30  * strength * jitter, 10, 80))
        hi = int(np.clip(120 * strength * jitter, 60, 250))
        edges = cv2.Canny(gray, lo, hi)
        # Dilate edges slightly so lines are visible at small sizes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        edge_color = np.zeros_like(base)
        edge_color[:, :] = (255, 255, 255)
        mask = edges > 0
        base[mask] = edge_color[mask]
        img = base
    elif tone == 'contour_inv':
        # Inverted Canny: white background, black contour lines — max contrast
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        lo = int(np.clip(20  * strength * jitter, 5,  60))
        hi = int(np.clip(100 * strength * jitter, 40, 200))
        edges = cv2.Canny(blur, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)
        inv = cv2.bitwise_not(edges)  # black lines on white
        img = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    elif tone == 'laplacian':
        # Laplacian edge detection blended with contrast-boosted original.
        #
        # The Laplacian is a second derivative, so it amplifies sensor noise
        # harder than it amplifies real edges. Fed the raw frame it produced
        # more response from the background grain than from the digit — the
        # object ended up less distinct than its surroundings, which is the
        # opposite of what the preset is for. Smoothing first is the standard
        # remedy (this is what "Laplacian of Gaussian" means); Canny already
        # blurs internally, which is why canny_overlay never had the problem.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap = np.clip(np.abs(lap) * 3.0 * strength * jitter, 0, 255).astype(np.uint8)
        boosted = cv2.convertScaleAbs(gray, alpha=1.4 * strength, beta=0)
        blended = cv2.addWeighted(boosted, 0.6, lap, 0.4, 0)
        img = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)
    elif tone == 'sobel':
        # Sobel gradient magnitude — emphasises horizontal+vertical edges
        # (digit strokes). Smoothed first for the same reason as 'laplacian':
        # on a grainy frame the raw gradient responds to noise everywhere and
        # the digit stops standing out from the background.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx ** 2 + sy ** 2)
        mag = np.clip(mag * 2.0 * strength * jitter, 0, 255).astype(np.uint8)
        # Blend with grayscale so background texture is preserved
        base = cv2.convertScaleAbs(gray, alpha=0.6)
        blended = cv2.addWeighted(base, 0.4, mag, 0.6, 0)
        img = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)
    elif tone == 'emboss':
        # Emboss convolution — makes raised/sunken digits cast a virtual shadow
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k = np.array([[-2, -1,  0],
                      [-1,  1,  1],
                      [ 0,  1,  2]], dtype=np.float32) * strength * jitter
        embossed = cv2.filter2D(gray.astype(np.float32), -1, k) + 128
        embossed = np.clip(embossed, 0, 255).astype(np.uint8)
        # The emboss kernel sums to zero, so every flat area — the object's
        # interior included — collapses to the same mid grey and only the
        # boundaries survive. Blending the relief back over the grayscale
        # source keeps the shading effect while the object still has a body.
        blended = cv2.addWeighted(gray, 0.5, embossed, 0.5, 0)
        img = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)
    elif tone == 'clahe_sharp':
        # Pipeline: bilateral denoise → CLAHE strong → unsharp mask
        # Best combination for low-contrast embossed numbers
        denoised = cv2.bilateralFilter(img, 9, 60, 60)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip = float(np.clip(5.0 * strength * jitter, 3.0, 10.0))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
        l2 = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        sigma = float(np.clip(1.2 * strength, 0.5, 2.5))
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigma)
        amount = float(np.clip(1.0 * strength * jitter, 0.5, 2.0))
        img = cv2.addWeighted(enhanced, 1.0 + amount, blur, -amount, 0)
    else:
        # fallback to slight brightness jitter so caller still gets variants
        img = cv2.convertScaleAbs(img, alpha=1.0 + 0.06 * strength * jitter, beta=int(6 * strength))

    return img


# A generated image is only worth training on if the annotated object is still
# there to be found. Presets suit different data: black-hat looks for sunken
# digits and finds nothing on raised ones, histogram equalisation can flatten a
# small object into its background. Rather than assume all 26 presets suit every
# project, each variant is checked before it is kept.
#
# The test is relative, not absolute. A photograph of a dark digit on a busy
# plate carries little contrast to begin with, so demanding a fixed amount
# would throw away good data. What is demanded is that the variant retain a
# share of whatever the source image had inside the box.
RETAIN_FRACTION = 0.35


def _region_separation(image, box):
    """
    How much there still is to see inside a box.

    Returns (internal contrast, internal edge energy), or None when the box is
    too small to measure. Both are reported because the presets split into two
    families: most keep tonal range, while the edge-map ones flatten tone on
    purpose and carry the object in the edges instead. A variant is judged on
    whichever of the two it preserves.

    This deliberately does not compare the box's average against its
    surroundings. A box drawn around a bright digit on a dark plate contains
    both, and a filter can shift them until the two averages coincide while the
    digit stays perfectly legible — three sound variants were thrown away that
    way before this measured the box's own content instead.
    """
    x, y, w, h = box
    height, width = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    # float64, not float32: cv2.Laplacian refuses a 32-bit source with a
    # 64-bit destination.
    grey = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float64)
    edges = np.abs(cv2.Laplacian(grey, cv2.CV_64F))
    return float(grey.std()), float(edges.mean())


def _boxes_from_regions(regions, width, height):
    """Annotation regions as integer pixel boxes, clamped to the frame."""
    boxes = []
    for region in regions or []:
        try:
            x = int(round(float(region.get('x', 0))))
            y = int(round(float(region.get('y', 0))))
            w = int(round(float(region.get('width', 0))))
            h = int(round(float(region.get('height', 0))))
        except (TypeError, ValueError):
            continue
        x, y = max(0, x), max(0, y)
        w, h = min(w, width - x), min(h, height - y)
        if w >= 4 and h >= 4:
            boxes.append((x, y, w, h))
    return boxes


def _object_survived(variant, boxes, baselines):
    """
    True when at least one annotated box still has something visible in it.

    A variant that fails this is not a slightly worse training image — it is an
    image whose label points at something no longer visible, which teaches the
    detector that the class looks like blank plate.
    """
    for box, baseline in zip(boxes, baselines):
        if baseline is None:
            return True          # unmeasurable: keep rather than guess
        after = _region_separation(variant, box)
        if after is None:
            return True
        if (after[0] >= baseline[0] * RETAIN_FRACTION
                or after[1] >= baseline[1] * RETAIN_FRACTION):
            return True
    return not boxes             # nothing annotated: nothing to lose


def _tone_seed(source_name, tone, index):
    """Stable per-variant seed so repeated runs reproduce the same images."""
    import zlib
    return zlib.crc32(f'{source_name}|{tone}|{index}'.encode('utf-8'))


def augment_color_images(name, source_filenames=None, tones=None,
                         variants_per_tone=3, strength=1.0,
                         require_all_annotated=True):
    """
    Create tone variants of annotated images, copying their annotations.

    By default only original (non-augmented) annotated images are used as
    sources; augmenting an augmented image would chain the file names and
    multiply near-identical data without adding information.
    """
    projects.get_project(name)
    summary = projects.refresh_stats(name)

    total = summary['total_images']
    annotated = summary['annotated_images']
    if require_all_annotated and total > 0 and annotated < total:
        raise ProjectError(
            f'Annotate all images first ({annotated}/{total} done), '
            'or disable the "require all annotated" option.'
        )

    if tones:
        tones = [str(t).strip().lower() for t in tones if str(t).strip()]
        tones = [t for t in tones if t in ALL_COLOR_TONES]
    if not tones:
        tones = list(ALL_COLOR_TONES)

    try:
        variants_per_tone = max(1, min(20, int(variants_per_tone)))
    except (TypeError, ValueError):
        variants_per_tone = 3
    try:
        strength = float(strength)
    except (TypeError, ValueError):
        strength = 1.0

    images_dir = projects.images_dir(name)

    if source_filenames:
        sources = []
        for filename in source_filenames:
            # One bad name should not abort the whole batch, which is what
            # letting safe_filename raise here used to do.
            try:
                path = images_dir / projects.safe_filename(filename)
            except ProjectError:
                continue
            if path.exists():
                sources.append(path)
    else:
        sources = []
        for entry in projects.list_images(name):
            if entry['annotated'] and not entry['augmented']:
                sources.append(images_dir / entry['filename'])

    if not sources:
        raise ProjectError('No annotated source images available for augmentation')

    planned = len(sources) * len(tones) * variants_per_tone
    if planned > MAX_GENERATED_IMAGES:
        raise ProjectError(
            f'That would generate {planned:,} images '
            f'({len(sources)} sources x {len(tones)} presets x {variants_per_tone} '
            f'variants), over the {MAX_GENERATED_IMAGES:,} limit. '
            'Choose fewer presets, fewer variants, or a subset of images.'
        )

    created, skipped = [], []
    dropped = {}
    for source in sources:
        img = imread(source)
        if img is None:
            skipped.append({'filename': source.name, 'reason': 'read_failed'})
            continue

        ann = projects.read_annotation(name, source.name)
        stem = source.stem[:MAX_STEM]

        # What the object's separation looks like before any filtering, so each
        # variant can be judged against its own source rather than a fixed bar.
        height, width = img.shape[:2]
        boxes = _boxes_from_regions(ann.get('regions'), width, height)
        baselines = [_region_separation(img, box) for box in boxes]

        for tone in tones:
            for index in range(variants_per_tone):
                stamp = datetime.now().strftime('%H%M%S%f')
                new_name = f'{stem}_aug_{tone}_{index + 1}_{stamp}.jpg'
                seed = _tone_seed(source.name, tone, index)

                try:
                    variant = apply_color_tone(img, tone, strength=strength, seed=seed)
                except Exception as exc:  # noqa: BLE001 - one bad preset must not abort the run
                    skipped.append({'filename': source.name,
                                    'reason': f'{tone}:{exc}'})
                    continue

                if not _object_survived(variant, boxes, baselines):
                    dropped[tone] = dropped.get(tone, 0) + 1
                    skipped.append({'filename': source.name,
                                    'reason': f'object_lost:{tone}'})
                    continue

                if not imwrite(images_dir / new_name, variant,
                               [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    skipped.append({'filename': source.name,
                                    'reason': f'write_failed:{tone}:{index + 1}'})
                    continue

                projects.write_annotation(name, new_name, {
                    'filename': new_name,
                    'regions': ann.get('regions', []),
                    'annotated': bool(ann.get('annotated')),
                    'width': ann.get('width'),
                    'height': ann.get('height'),
                    # A copy belongs to whichever upload its source came from,
                    # so filtering the gallery by batch shows a photograph
                    # together with everything made from it.
                    'batch': ann.get('batch'),
                    'imported_at': ann.get('imported_at'),
                    'augmented': True,
                    'augmentation': {
                        'type': 'color_tone',
                        'tone': tone,
                        'strength': strength,
                        'source_image': source.name,
                    },
                    'updated_at': datetime.now().isoformat(),
                })
                created.append({'filename': new_name, 'source': source.name, 'tone': tone})

    # A run can add hundreds of files, so the index is rebuilt once rather
    # than patched per image.
    projects.rebuild_index(name)
    projects.refresh_stats(name)
    message = f'Created {len(created)} images from {len(sources)} sources'
    if dropped:
        worst = ', '.join(f'{tone} ({count})' for tone, count
                          in sorted(dropped.items(), key=lambda kv: -kv[1])[:4])
        message += (f'. Dropped {sum(dropped.values())} where the filter hid the '
                    f'annotated object: {worst}')

    return {
        'message': message,
        'planned': planned,
        'dropped_by_tone': dropped,
        'created_count': len(created),
        'source_count': len(sources),
        'tones': tones,
        'variants_per_tone': variants_per_tone,
        'created': created[:300],
        'skipped': skipped[:100],
    }
