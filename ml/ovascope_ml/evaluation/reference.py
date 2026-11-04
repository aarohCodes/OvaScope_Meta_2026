"""Reference-standard access for EVALUATION ONLY.

This is the only module that reads test-split labels or test reference masks.
Inference code (`ovascope_ml.inference`) must never import it; a test enforces this.
Oracle experiments that use test masks write to `eval_only/` directories and are
reported separately as upper bounds.
"""

from __future__ import annotations

import numpy as np

from ovascope_ml import paths
from ovascope_ml.data.imaging import load_mask_file
from ovascope_ml.data.manifest import Manifest
from ovascope_ml.training.common import CLASS_NAMES


def test_labels(m: Manifest) -> dict[str, int]:
    return {r.image_id: CLASS_NAMES.index(r.label_normalized) for r in m.by_split("test")}


def test_reference_masks(m: Manifest) -> dict[str, np.ndarray | None]:
    root = paths.mmotu_root()
    out: dict[str, np.ndarray | None] = {}
    for r in m.by_split("test"):
        out[r.image_id] = load_mask_file(root / r.mask_path_if_available) if r.mask_path_if_available else None
    return out


def reference_mask(m: Manifest, image_id: str) -> np.ndarray | None:
    rec = next(r for r in m.records if r.image_id == image_id)
    if rec.mask_path_if_available is None:
        return None
    return load_mask_file(paths.mmotu_root() / rec.mask_path_if_available)
