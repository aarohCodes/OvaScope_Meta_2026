"""Render overlay artifacts as source-resolution RGBA PNGs.

Overlays are produced at the exact pixel size of the uploaded image, so the
viewer can draw them with the same transform as the image (no resampling drift).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage

TEAL = (45, 212, 191)
VIOLET = (167, 139, 250)
CYAN = (34, 211, 238)


def _png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def mask_png(mask: np.ndarray) -> bytes:
    edge = mask & ~ndimage.binary_erosion(mask, iterations=2)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*TEAL, 96)
    rgba[edge] = (*TEAL, 255)
    return _png(rgba)


def heat_png(values: np.ndarray, colour: tuple[int, int, int] = CYAN, max_alpha: int = 170) -> bytes:
    v = np.clip(values.astype(np.float32), 0, 1)
    rgba = np.zeros((*v.shape, 4), dtype=np.uint8)
    rgba[..., :3] = colour
    rgba[..., 3] = (v * max_alpha).astype(np.uint8)
    return _png(rgba)


def diverging_png(values: np.ndarray, max_alpha: int = 180) -> bytes:
    """Positive -> teal, negative -> violet; alpha proportional to |value| / max|value|."""
    v = values.astype(np.float32)
    m = float(np.abs(v).max()) or 1.0
    a = np.clip(np.abs(v) / m, 0, 1)
    rgba = np.zeros((*v.shape, 4), dtype=np.uint8)
    pos = v >= 0
    rgba[pos, :3] = TEAL
    rgba[~pos, :3] = VIOLET
    rgba[..., 3] = (a * max_alpha).astype(np.uint8)
    return _png(rgba)
