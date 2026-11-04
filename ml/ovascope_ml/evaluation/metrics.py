"""Evaluation metrics.

All PPV/NPV values are within the evaluated, enriched cohort (class prevalence is
reported alongside); they do not transfer to screening populations.
Bootstrap intervals resample images. Patient identifiers are unavailable, so
within-patient correlation is ignored and intervals are likely too narrow.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def _safe_ap(y: np.ndarray, s: np.ndarray) -> float | None:
    if y.sum() == 0:
        return None
    return float(average_precision_score(y, s))


def _div(a: float, b: float) -> float | None:
    return None if b == 0 else float(a / b)


def macro_auroc(y: np.ndarray, scores: np.ndarray) -> float | None:
    raw = [_safe_auc((y == k).astype(int), scores[:, k]) for k in range(scores.shape[1])]
    vals = [v for v in raw if v is not None]
    return float(np.mean(vals)) if vals else None


def balanced_accuracy(y: np.ndarray, scores: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, scores.argmax(1)))


def ece_top_label(y: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    conf = scores.max(1)
    correct = (scores.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if sel.sum() == 0:
            continue
        acc, mc = float(correct[sel].mean()), float(conf[sel].mean())
        ece += sel.mean() * abs(acc - mc)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()), "mean_score": mc, "accuracy": acc})
    return float(ece), bins


def multiclass_metrics(y: np.ndarray, scores: np.ndarray, class_names: list[str]) -> dict[str, object]:
    pred = scores.argmax(1)
    k = len(class_names)
    cm = confusion_matrix(y, pred, labels=list(range(k)))
    per_class = []
    for c in range(k):
        yt = (y == c).astype(int)
        tp = int(cm[c, c])
        fn = int(cm[c].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        per_class.append(
            {
                "label": class_names[c],
                "n": int(yt.sum()),
                "prevalence": float(yt.mean()),
                "sensitivity": _div(tp, tp + fn),
                "specificity": _div(tn, tn + fp),
                "ppv": _div(tp, tp + fp),
                "npv": _div(tn, tn + fn),
                "auroc_ovr": _safe_auc(yt, scores[:, c]),
                "auprc_ovr": _safe_ap(yt, scores[:, c]),
            }
        )
    eps = 1e-12
    onehot = np.eye(k)[y]
    ece, rel = ece_top_label(y, scores)
    return {
        "n": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float((pred == y).mean()),
        "macro_auroc_ovr": macro_auroc(y, scores),
        "per_class": per_class,
        "confusion_matrix": {"labels": class_names, "rows_true_cols_pred": cm.tolist()},
        "calibration": {
            "top_label_ece_10bin": ece,
            "reliability_bins": rel,
            "multiclass_brier": float(((scores - onehot) ** 2).sum(1).mean()),
            "nll": float(-np.log(scores[np.arange(len(y)), y] + eps).mean()),
        },
    }


def curve_points(x: np.ndarray, y: np.ndarray, max_points: int = 200) -> list[list[float]]:
    if len(x) > max_points:
        idx = np.unique(np.linspace(0, len(x) - 1, max_points).round().astype(int))
        x, y = x[idx], y[idx]
    return [[float(a), float(b)] for a, b in zip(x, y, strict=True)]


def binary_metrics(yb: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (yb == 1)).sum())
    fp = int(((pred == 1) & (yb == 0)).sum())
    tn = int(((pred == 0) & (yb == 0)).sum())
    fn = int(((pred == 0) & (yb == 1)).sum())
    out: dict[str, object] = {
        "n": int(len(yb)),
        "n_positive": int(yb.sum()),
        "prevalence": float(yb.mean()),
        "auroc": _safe_auc(yb, s),
        "auprc": _safe_ap(yb, s),
        "threshold": float(threshold),
        "threshold_source": "development out-of-fold (Youden J), frozen before test",
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "sensitivity": _div(tp, tp + fn),
        "specificity": _div(tn, tn + fp),
        "ppv": _div(tp, tp + fp),
        "npv": _div(tn, tn + fn),
    }
    if len(np.unique(yb)) == 2:
        fpr, tpr, _ = roc_curve(yb, s)
        prec, rec, _ = precision_recall_curve(yb, s)
        out["roc_curve"] = curve_points(fpr, tpr)
        out["pr_curve"] = curve_points(rec[::-1], prec[::-1])
    return out


def risk_coverage(y: np.ndarray, scores: np.ndarray, n_points: int = 20) -> list[dict[str, float]]:
    conf = scores.max(1)
    order = np.argsort(-conf)
    pts = []
    for cov in np.linspace(0.1, 1.0, n_points):
        keep = order[: max(1, round(cov * len(y)))]
        pts.append(
            {
                "coverage": float(len(keep) / len(y)),
                "accuracy": float((scores[keep].argmax(1) == y[keep]).mean()),
                "balanced_accuracy": float(balanced_accuracy_score(y[keep], scores[keep].argmax(1))),
            }
        )
    return pts


def abstention_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    keep = scores.max(1) >= threshold
    out: dict[str, object] = {
        "threshold": float(threshold),
        "threshold_source": "development out-of-fold scores (target 90% coverage), frozen before test",
        "coverage": float(keep.mean()),
        "n_retained": int(keep.sum()),
        "n_abstained": int((~keep).sum()),
        "accuracy_all": float((scores.argmax(1) == y).mean()),
    }
    if keep.any():
        out["accuracy_retained"] = float((scores[keep].argmax(1) == y[keep]).mean())
        out["balanced_accuracy_retained"] = float(balanced_accuracy_score(y[keep], scores[keep].argmax(1)))
    if (~keep).any():
        out["accuracy_abstained"] = float((scores[~keep].argmax(1) == y[~keep]).mean())
    return out


Metric = Callable[[np.ndarray, np.ndarray], float | None]


def bootstrap_ci(metric: Metric, y: np.ndarray, s: np.ndarray, n: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        v = metric(y[idx], s[idx])
        if v is not None:
            vals.append(v)
    point = metric(y, s)
    if not vals:
        return {"point": point, "ci95": None, "n_resamples": 0}
    return {
        "point": point,
        "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        "n_resamples": len(vals),
        "method": "image-level percentile bootstrap",
    }


def paired_bootstrap_diff(
    metric: Metric, y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, n: int, seed: int
) -> dict[str, object]:
    """Distribution of metric(a) - metric(b) over shared image resamples."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        a, b = metric(y[idx], s_a[idx]), metric(y[idx], s_b[idx])
        if a is not None and b is not None:
            diffs.append(a - b)
    pa, pb = metric(y, s_a), metric(y, s_b)
    return {
        "difference": None if pa is None or pb is None else float(pa - pb),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if diffs else None,
        "fraction_resamples_a_better": float(np.mean(np.array(diffs) > 0)) if diffs else None,
        "method": "paired image-level bootstrap",
    }


def summarize_dice(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
    a = np.asarray(values)
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
    }
