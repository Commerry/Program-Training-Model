"""
What augmentation a run should use, decided from the project's own data.

Two separate things go by the name "augmentation" here and they are worth
keeping apart, because they solve different problems and one of them was
running unsupervised.

**During training**, ultralytics augments every image on the fly, differently
each epoch. Nothing in this application passed it any settings, so it used its
own defaults, which include mirroring half the images. For a project whose
classes are digits that is not a small inefficiency: a mirrored 2 is not a 2,
and the label still says it is, so a share of every epoch teaches the model
something false. On-the-fly augmentation is otherwise the better kind -- it
costs no disk and gives a fresh variant each epoch rather than a fixed set.

**Before training**, the colour presets in services/augment.py write new image
files. These are worth generating only for the transformations ultralytics does
not do at all: edge maps, adaptive thresholds, morphological top-hat, CLAHE.
Re-generating brightness and colour variants as files duplicates what the
trainer already does every epoch, and multiplies the time each epoch takes by
the number of copies.
"""

from services import projects

# Presets that change the image in a way training does not already do for free.
# Deliberately not the whole list of 26: hsv jitter, brightness and contrast are
# applied on the fly every epoch, so writing them to disk adds epoch time
# without adding information.
STRUCTURAL_PRESETS = [
    'clahe',            # local contrast, for a plate lit unevenly
    'clahe_strong',
    'clahe_sharp',
    'adaptive_thresh',  # binarised, for engraved or embossed characters
    'tophat',           # bright detail on a darker background
    'blackhat',         # the same for sunken detail
    'canny_overlay',    # outlines drawn over the frame
    'contour_inv',
    'sobel',
    'laplacian',
    'unsharp',
    'bilateral',        # denoise, keeping edges
]

# How many of the above to use at the default setting. All twelve on a large
# project is a lot of files; these are the ones that carry the most information
# for the plate-and-character case this tool is usually pointed at.
DEFAULT_PRESETS = ['clahe', 'clahe_sharp', 'adaptive_thresh', 'tophat',
                   'canny_overlay', 'unsharp']


def classes_are_orientation_sensitive(class_names):
    """
    Whether mirroring an image would change what its labels mean.

    A single character is the clear case: 2, 5, R and b all become something
    that is not themselves when flipped, while the label travels with the image
    unchanged. Words and multi-character names are treated the same way for the
    same reason. Only when nothing in the list is text-like -- 'cat', 'bolt',
    'crack' -- is mirroring safe, and that is the case this returns False for.
    """
    names = [str(n).strip() for n in (class_names or []) if str(n).strip()]
    if not names:
        return False

    # Any single character, digit or letter, is orientation-bearing.
    if any(len(n) == 1 and (n.isdigit() or n.isalpha()) for n in names):
        return True

    # A set of names that are all numbers -- '10', '25', '100' -- is a readout
    # being transcribed, and the same argument applies.
    if all(n.replace('.', '', 1).isdigit() for n in names):
        return True

    return False


def recommend(name):
    """
    Augmentation settings for this project, and why.

    Returned rather than applied so the training screen can show what will
    happen and let it be overridden: the reasoning below is a good default, not
    a fact about the user's data that this code can be certain of.
    """
    summary = projects.dataset_summary(name)
    class_names = summary.get('classes') or []
    sensitive = classes_are_orientation_sensitive(class_names)

    settings = {
        # Colour and exposure: kept, and applied fresh every epoch.
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        # Geometry that does not change what an object is.
        'translate': 0.1,
        'scale': 0.5,
        'degrees': 0.0,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.0 if sensitive else 0.5,
        'mosaic': 1.0,
        'erasing': 0.4,
    }

    reasons = []
    if sensitive:
        reasons.append(
            'Left-right mirroring is off because every class is a character or '
            'a number, and a mirrored 2 is not a 2 while its label still says '
            'it is. Ultralytics mirrors half of every epoch by default.')
    else:
        reasons.append(
            'Left-right mirroring is on: no class here reads as text, so a '
            'mirrored object is still the same object.')

    annotated = summary.get('annotated_images', 0)
    if annotated and annotated < 50:
        reasons.append(
            f'Only {annotated} annotated images. Generating filtered copies '
            'before training is worth doing here; on-the-fly augmentation '
            'alone has little to vary.')

    return {
        'settings': settings,
        'orientation_sensitive': sensitive,
        'classes': class_names,
        'annotated_images': annotated,
        'reasons': reasons,
        'suggested_presets': DEFAULT_PRESETS,
        'available_presets': STRUCTURAL_PRESETS,
    }


def sanitise(requested):
    """
    A caller-supplied augmentation dict, reduced to values ultralytics accepts.

    Anything absent keeps the recommendation's value; anything out of range is
    clamped rather than refused, because a slider that reports 1.4 should not
    fail a run that took minutes to reach this point.
    """
    limits = {
        'hsv_h': (0.0, 1.0), 'hsv_s': (0.0, 1.0), 'hsv_v': (0.0, 1.0),
        'degrees': (0.0, 180.0), 'translate': (0.0, 1.0), 'scale': (0.0, 1.0),
        'shear': (0.0, 180.0), 'perspective': (0.0, 0.001),
        'flipud': (0.0, 1.0), 'fliplr': (0.0, 1.0),
        'mosaic': (0.0, 1.0), 'erasing': (0.0, 0.9),
    }
    clean = {}
    for key, (low, high) in limits.items():
        if key not in (requested or {}):
            continue
        try:
            clean[key] = min(max(float(requested[key]), low), high)
        except (TypeError, ValueError):
            continue
    return clean


# ── How the images reach the GPU ────────────────────────────────────────────

# Kept low rather than at ultralytics' default of 8. This worker is itself a
# spawned subprocess, which is the arrangement where the dataloader used to
# deadlock on Windows. Measured through that exact path -- the application's
# own subprocess, on Windows, with the ultralytics in requirements.txt -- a run
# at 4 completes normally and produces a model indistinguishable from one
# trained at 0. A small number gets most of the benefit with less to go wrong,
# and 0 remains available if the deadlock ever returns.
DEFAULT_WORKERS = 4

# Holding the decoded images in memory is NOT offered, and that is a
# measurement rather than caution.
#
# Trained twice on the same 20 images with the same seed and epochs, differing
# only in this setting, both runs reported mAP50 = 0.995. Asked about ten
# images from the same generator that neither had seen:
#
#     cache off  ->  0.29, 0.21, 0.16, 0.21, 0.15, 0.27, 0.19, 0.09, 0.33, 0.20
#     cache ram  ->  0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
#
# The cached run produces a model that predicts nothing while reporting a
# perfect score. That is the exact failure this application already shipped
# once -- a run that completes, reports well, and yields weights that detect
# nothing -- so the setting stays off. It remains reachable by passing cache
# explicitly, for anyone who wants to re-measure it on a newer ultralytics.
CACHE_IS_SAFE = False


def loader_settings(image_count, img_size, workers=None, cache=None):
    """
    How many loader processes to use, and whether to hold images in memory.

    Both were previously pinned: workers at 0 and cache off. On a slow card
    neither matters, because the GPU is what is being waited on. On a fast one
    a single process decoding JPEGs becomes the bottleneck and the card idles
    between batches, which looks exactly like a slow GPU and gives no hint of
    the real cause -- so workers is now chosen rather than pinned.

    `image_count` and `img_size` are accepted for the caching decision and are
    unused while caching stays off; they are kept so re-enabling it does not
    mean changing every caller.

    Returns (workers, cache) where cache is 'ram' or False.
    """
    try:
        workers = max(0, min(16, int(workers)))
    except (TypeError, ValueError):
        workers = DEFAULT_WORKERS

    if cache is None:
        cache = 'ram' if CACHE_IS_SAFE else False
    elif cache in (True, 'ram', 'disk'):
        cache = 'ram' if cache is True else cache
    else:
        cache = False

    return workers, cache


def cache_estimate(image_count, img_size):
    """Bytes an in-memory cache of this set would need, for the UI to show."""
    try:
        return int(image_count) * int(img_size) * int(img_size) * 3
    except (TypeError, ValueError):
        return 0
