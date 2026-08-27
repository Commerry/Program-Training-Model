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
