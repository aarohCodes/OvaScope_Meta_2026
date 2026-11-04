"""Aspect-preserving letterbox transform with exact inverse mapping.

Every model input, patch grid, mask and overlay goes through this one transform,
so coordinates can be mapped back to the source image without drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Letterbox:
    src_w: int
    src_h: int
    size: int

    @property
    def scale(self) -> float:
        return self.size / max(self.src_w, self.src_h)

    @property
    def new_w(self) -> int:
        return max(1, round(self.src_w * self.scale))

    @property
    def new_h(self) -> int:
        return max(1, round(self.src_h * self.scale))

    @property
    def pad_x(self) -> int:
        return (self.size - self.new_w) // 2

    @property
    def pad_y(self) -> int:
        return (self.size - self.new_h) // 2

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        assert rgb.shape[:2] == (self.src_h, self.src_w), "image does not match letterbox source size"
        im = Image.fromarray(rgb).resize((self.new_w, self.new_h), Image.Resampling.BICUBIC)
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[self.pad_y : self.pad_y + self.new_h, self.pad_x : self.pad_x + self.new_w] = np.asarray(im)
        return canvas

    def apply_mask(self, mask: np.ndarray) -> np.ndarray:
        assert mask.shape == (self.src_h, self.src_w), "mask does not match letterbox source size"
        im = Image.fromarray(mask.astype(np.uint8) * 255).resize((self.new_w, self.new_h), Image.Resampling.NEAREST)
        canvas = np.zeros((self.size, self.size), dtype=bool)
        canvas[self.pad_y : self.pad_y + self.new_h, self.pad_x : self.pad_x + self.new_w] = np.asarray(im) > 127
        return canvas

    def valid_region(self) -> np.ndarray:
        m = np.zeros((self.size, self.size), dtype=bool)
        m[self.pad_y : self.pad_y + self.new_h, self.pad_x : self.pad_x + self.new_w] = True
        return m

    def to_model(self, x: float, y: float) -> tuple[float, float]:
        return x * self.new_w / self.src_w + self.pad_x, y * self.new_h / self.src_h + self.pad_y

    def to_source(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.pad_x) * self.src_w / self.new_w, (y - self.pad_y) * self.src_h / self.new_h

    def model_map_to_source(self, model_map: np.ndarray) -> np.ndarray:
        """Resample a (size, size) or coarser square map back to source resolution.

        A coarse map (e.g. a 32x32 patch grid) is first upsampled to the model canvas,
        then the letterbox padding is cropped away and the content resized to source size.
        """
        arr = model_map.astype(np.float32)
        if arr.shape != (self.size, self.size):
            arr = np.asarray(Image.fromarray(arr).resize((self.size, self.size), Image.Resampling.BILINEAR))
        crop = arr[self.pad_y : self.pad_y + self.new_h, self.pad_x : self.pad_x + self.new_w]
        out = Image.fromarray(np.ascontiguousarray(crop)).resize((self.src_w, self.src_h), Image.Resampling.BILINEAR)
        return np.asarray(out)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x1, y1, x2, y2) inclusive-exclusive pixel box of a boolean mask, or None."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_box(
    box: tuple[float, float, float, float], margin: float, width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    dw, dh = (x2 - x1) * margin, (y2 - y1) * margin
    return (
        max(0, int(np.floor(x1 - dw))),
        max(0, int(np.floor(y1 - dh))),
        min(width, int(np.ceil(x2 + dw))),
        min(height, int(np.ceil(y2 + dh))),
    )


def dice_iou(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    p, r = pred.astype(bool), ref.astype(bool)
    inter = float(np.logical_and(p, r).sum())
    ps, rs = float(p.sum()), float(r.sum())
    union = ps + rs - inter
    dice = 1.0 if ps + rs == 0 else 2 * inter / (ps + rs)
    iou = 1.0 if union == 0 else inter / union
    return dice, iou
