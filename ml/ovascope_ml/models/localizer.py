"""Automatic candidate-region localiser on frozen DINOv2 patch tokens.

A linear (logistic) head scores each 14x14-pixel patch of the letterboxed
448x448 input. It is trained on development-split reference masks only. The
largest high-scoring connected component yields a box prompt for SAM.

This is a localisation prior, not a diagnosis, and it can be wrong or empty.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import ndimage

from ovascope_ml.models.geometry import Letterbox, mask_bbox
from ovascope_ml.models.heads import LinearHead


@dataclass(frozen=True)
class LocalizerParams:
    threshold: float = 0.5
    smooth_sigma: float = 0.75  # in patch units
    min_component_patches: int = 4

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LocalizerParams:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass
class LocalizationResult:
    status: Literal["candidate_found", "no_candidate_region"]
    prob_grid: np.ndarray  # (g, g) smoothed patch scores in model space
    coarse_mask: np.ndarray | None  # (H, W) bool in source space
    box: tuple[int, int, int, int] | None  # (x1, y1, x2, y2) source pixels
    max_score: float
    component_patches: int


def patch_targets(mask_model: np.ndarray, grid: int) -> np.ndarray:
    """Fraction of foreground pixels per patch of a (size, size) model-space mask."""
    size = mask_model.shape[0]
    p = size // grid
    frac = mask_model[: grid * p, : grid * p].reshape(grid, p, grid, p).mean(axis=(1, 3))
    return frac


class PatchLocalizer:
    def __init__(self, head: LinearHead, params: LocalizerParams) -> None:
        assert head.kind == "sigmoid"
        self.head = head
        self.params = params

    def prob_grid(self, patches: np.ndarray, letterbox: Letterbox) -> np.ndarray:
        g = patches.shape[0]
        probs = self.head.scores(patches.reshape(g * g, -1).astype(np.float32)).reshape(g, g)
        if self.params.smooth_sigma > 0:
            probs = ndimage.gaussian_filter(probs, sigma=self.params.smooth_sigma, mode="nearest")
        valid = letterbox.valid_region()
        p = letterbox.size // g
        valid_grid = valid[: g * p, : g * p].reshape(g, p, g, p).mean(axis=(1, 3)) > 0.5
        return np.where(valid_grid, probs, 0.0)

    def localize(self, patches: np.ndarray, letterbox: Letterbox) -> LocalizationResult:
        grid = self.prob_grid(patches, letterbox)
        return self.from_grid(grid, letterbox)

    def from_grid(self, grid: np.ndarray, letterbox: Letterbox) -> LocalizationResult:
        return region_from_grid(grid, letterbox, self.params)


def region_from_grid(grid: np.ndarray, letterbox: Letterbox, params: LocalizerParams) -> LocalizationResult:
    """Largest-mass component above threshold -> source-space coarse mask and box."""
    above = grid >= params.threshold
    labels, n = ndimage.label(above)
    max_score = float(grid.max()) if grid.size else 0.0
    if n == 0:
        return LocalizationResult("no_candidate_region", grid, None, None, max_score, 0)
    masses = ndimage.sum(grid, labels, index=np.arange(1, n + 1))
    best = int(np.argmax(masses)) + 1
    comp = labels == best
    n_patches = int(comp.sum())
    if n_patches < params.min_component_patches:
        return LocalizationResult("no_candidate_region", grid, None, None, max_score, n_patches)
    src = letterbox.model_map_to_source(comp.astype(np.float32)) >= 0.5
    box = mask_bbox(src)
    if box is None:
        return LocalizationResult("no_candidate_region", grid, None, None, max_score, n_patches)
    return LocalizationResult("candidate_found", grid, src, box, max_score, n_patches)
