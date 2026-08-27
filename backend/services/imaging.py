"""
Image helpers that behave correctly on Windows with non-ASCII paths.

cv2.imread / cv2.imwrite pass the path to the C++ layer as a byte string using
the ANSI code page, so any project or file name outside that code page silently
fails and returns None. Reading the bytes ourselves and decoding from memory
avoids the problem entirely.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def imread(path):
    """Read an image as BGR. Returns None if it cannot be decoded."""
    path = Path(path)
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite(path, image, params=None):
    """Write a BGR image. Returns True on success."""
    path = Path(path)
    ext = path.suffix or '.jpg'
    ok, buf = cv2.imencode(ext, image, params or [])
    if not ok:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
    except OSError:
        return False
    return True


def image_size(path):
    """
    Return (width, height) without decoding pixel data.

    Pillow reads only the header, which makes listing thousands of images
    cheap enough to do on every request.

    A DecompressionBombError is deliberately not retried with a full cv2
    decode: that warning exists precisely because decoding the image would
    exhaust memory, and falling back to cv2 would cause the very problem
    Pillow refused to cause.
    """
    try:
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Image.DecompressionBombError:
        return None
    except Exception:
        img = imread(path)
        if img is None:
            return None
        return int(img.shape[1]), int(img.shape[0])
