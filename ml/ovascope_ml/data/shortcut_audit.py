"""Shortcut audit: can trivial, non-anatomical image properties predict the label?

If frame geometry (a proxy for device/preset/crop) or colour content (colour
Doppler and coloured burned-in annotations) predicts the category well, any
image model may be exploiting the same cues. The probe below quantifies that
risk; it does not remove it.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

from ovascope_ml.data.validation import ImageRow

FEATURES = ("width", "height", "aspect", "log_area", "chroma_fraction")


def _matrix(rows: Sequence[ImageRow]) -> np.ndarray:
    out = []
    for r in rows:
        w, h = float(r.width or 0), float(r.height or 0)
        out.append([w, h, w / h if h else 0.0, np.log(max(w * h, 1.0)), r.chroma_fraction or 0.0])
    return np.asarray(out, dtype=np.float64)


def metadata_probe(
    rows: Sequence[ImageRow],
    labels: Sequence[int],
    folds: np.ndarray,
    test_rows: Sequence[ImageRow],
    test_labels: Sequence[int],
    seed: int = 0,
) -> dict[str, object]:
    """Grouped-CV and held-out balanced accuracy of a metadata-only classifier."""
    x, y = _matrix(rows), np.asarray(labels)
    oof = np.zeros(len(y), dtype=int)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, random_state=seed)
        clf.fit(x[tr], y[tr])
        oof[va] = clf.predict(x[va])
    final = HistGradientBoostingClassifier(max_depth=3, max_iter=150, random_state=seed)
    final.fit(x, y)
    xt, yt = _matrix(test_rows), np.asarray(test_labels)
    test_pred = final.predict(xt)

    # Binary view for the secondary target (class 7 vs rest).
    yb, ytb = (y == 7).astype(int), (yt == 7).astype(int)
    oofb = np.zeros(len(y), dtype=int)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, random_state=seed, class_weight="balanced")
        clf.fit(x[tr], yb[tr])
        oofb[va] = clf.predict(x[va])
    finalb = HistGradientBoostingClassifier(max_depth=3, max_iter=150, random_state=seed, class_weight="balanced")
    finalb.fit(x, yb)

    return {
        "features": list(FEATURES),
        "model": "HistGradientBoostingClassifier(max_depth=3, max_iter=150)",
        "chance_balanced_accuracy_mmotu8": 1.0 / len(np.unique(y)),
        "mmotu8": {
            "dev_grouped_cv_balanced_accuracy": float(balanced_accuracy_score(y, oof)),
            "test_balanced_accuracy": float(balanced_accuracy_score(yt, test_pred)),
        },
        "hgs_vs_other": {
            "dev_grouped_cv_balanced_accuracy": float(balanced_accuracy_score(yb, oofb)),
            "test_balanced_accuracy": float(balanced_accuracy_score(ytb, finalb.predict(xt))),
        },
        "interpretation": (
            "Balanced accuracy of a classifier that sees only frame size, aspect ratio and "
            "colour-pixel fraction. Values well above chance mean acquisition/annotation "
            "artefacts carry label information that image models can also exploit."
        ),
    }


def per_class_artifact_table(rows: Sequence[ImageRow], class_names: Sequence[str]) -> list[dict[str, object]]:
    by_label: dict[str, list[ImageRow]] = collections.defaultdict(list)
    for r in rows:
        by_label[r.label].append(r)
    table = []
    for name in class_names:
        rs = by_label.get(name, [])
        chroma = np.array([r.chroma_fraction or 0.0 for r in rs]) if rs else np.zeros(0)
        sizes = collections.Counter(
            f"{round((r.width or 0) / 20) * 20}x{round((r.height or 0) / 20) * 20}" for r in rs
        )
        table.append(
            {
                "label": name,
                "n": len(rs),
                "fraction_with_colour_pixels_gt_1pct": float((chroma > 0.01).mean()) if rs else None,
                "median_colour_pixel_fraction": float(np.median(chroma)) if rs else None,
                "top_frame_size_clusters_20px": sizes.most_common(3),
            }
        )
    return table
