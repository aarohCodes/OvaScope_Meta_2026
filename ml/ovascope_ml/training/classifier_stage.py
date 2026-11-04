"""Stage 5: classifier heads, model selection, calibration and frozen thresholds.

Uses development labels only. Test outputs are scores WITHOUT labels; labels are
joined later by `ovascope-ml evaluate`. Everything selected here is written to
selection.json before any test metric exists.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler

from ovascope_ml.models.heads import LinearHead, fit_temperature
from ovascope_ml.training import localizer_stage
from ovascope_ml.training.common import (
    CLASS_NAMES,
    HGS_INDEX,
    FeatureStore,
    dev_view,
    experiment_dir,
    load_manifest,
    log,
    stage_dir,
    test_inputs,
)

# name -> (feature blocks, role)
VARIANTS: dict[str, tuple[tuple[str, ...], str]] = {
    "majority": ((), "baseline"),
    "handcrafted": (("handcrafted",), "baseline"),
    "resnet50_imagenet": (("resnet50",), "comparator"),
    "dinov2_whole": (("dino_global",), "meta"),
    "dinov2_roi_coarse": (("dino_global", "roi_coarse"), "ablation"),
    "dinov2_roi_sam": (("dino_global", "roi_sam"), "meta"),
    "oracle_dinov2_roi_gt": (("dino_global", "roi_gt"), "oracle"),
}
DEPLOYABLE = ("dinov2_whole", "dinov2_roi_sam")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_lr(
    x: np.ndarray, y: np.ndarray, c: float, class_weight: str | None, seed: int
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(x)
    lr = LogisticRegression(C=c, max_iter=5000, class_weight=class_weight, random_state=seed)
    lr.fit(scaler.transform(x), y)
    if list(lr.classes_) != list(range(len(CLASS_NAMES))):
        raise RuntimeError("a class is missing from a training fold; cannot build an 8-class head")
    return scaler, lr


def _blocks(cfg: dict[str, Any], names: tuple[str, ...], ids: list[str], split: str) -> np.ndarray:
    feats = FeatureStore(stage_dir(cfg, "features"))
    roi = FeatureStore(stage_dir(cfg, "roi"))
    roi_eval = FeatureStore(stage_dir(cfg, "roi") / "eval_only")
    arrs = []
    for n in names:
        if n == "roi_gt":
            arrs.append(roi.rows("roi_gt_dev", ids) if split == "dev" else roi_eval.rows("roi_gt_test", ids))
        elif n.startswith("roi_"):
            arrs.append(roi.rows(n, ids))
        else:
            arrs.append(feats.rows(n, ids))
    return np.concatenate(arrs, axis=1).astype(np.float64)


def run(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    out_dir = experiment_dir(cfg)
    sel_path = out_dir / "selection.json"
    if sel_path.exists() and not force:
        log("classifiers: cached")
        return json.loads(sel_path.read_text(encoding="utf-8"))
    m = load_manifest(cfg)
    dv, ti = dev_view(m), test_inputs(m)
    ids, folds = localizer_stage.folds_for(cfg)
    assert ids == dv.image_ids
    y = dv.labels
    cc = cfg["classifier"]
    seed = cfg["seed"]
    heads_dir = out_dir / "heads"
    heads_dir.mkdir(exist_ok=True)

    oof_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}
    fallback_dev: dict[str, np.ndarray] = {}
    fallback_test: dict[str, np.ndarray] = {}
    selection: dict[str, Any] = {"variants": {}}

    # Majority-class baseline: development prior.
    prior = np.bincount(y, minlength=len(CLASS_NAMES)) / len(y)
    oof_scores["majority"] = np.tile(prior, (len(y), 1))
    test_scores["majority"] = np.tile(prior, (len(ti.image_ids), 1))
    fallback_dev["majority"] = np.zeros(len(y), dtype=bool)
    fallback_test["majority"] = np.zeros(len(ti.image_ids), dtype=bool)
    selection["variants"]["majority"] = {
        "role": "baseline",
        "predicts": CLASS_NAMES[int(prior.argmax())],
        "dev_oof_balanced_accuracy": float(balanced_accuracy_score(y, oof_scores["majority"].argmax(1))),
    }

    whole_oof_logits = whole_test_logits = None
    whole_T = 1.0
    for name in [v for v in VARIANTS if v != "majority"]:
        blocks, role = VARIANTS[name]
        x_dev = _blocks(cfg, blocks, dv.image_ids, "dev")
        x_test = _blocks(cfg, blocks, ti.image_ids, "test")
        ok_dev = ~np.isnan(x_dev).any(axis=1)
        ok_test = ~np.isnan(x_test).any(axis=1)
        grid = []
        best = None
        for c in cc["C_grid"]:
            logits = np.zeros((len(y), len(CLASS_NAMES)))
            for f in np.unique(folds):
                tr = (folds != f) & ok_dev
                va = (folds == f) & ok_dev
                sc, lr = _fit_lr(x_dev[tr], y[tr], c, cc["class_weight"], seed)
                logits[va] = lr.decision_function(sc.transform(x_dev[va]))
            if not ok_dev.all():  # fallback rows use the frozen whole-image OOF logits
                assert whole_oof_logits is not None
                logits[~ok_dev] = whole_oof_logits[~ok_dev]
            bacc = float(balanced_accuracy_score(y, logits.argmax(1)))
            grid.append({"C": c, "dev_oof_balanced_accuracy": bacc})
            if best is None or bacc > best[1] + 1e-12:
                best = (c, bacc, logits)
        c_best, bacc_best, oof_logits = best  # type: ignore[misc]
        sc, lr = _fit_lr(x_dev[ok_dev], y[ok_dev], c_best, cc["class_weight"], seed)
        head = LinearHead.from_sklearn(sc, lr, CLASS_NAMES)
        t = fit_temperature(oof_logits[ok_dev], y[ok_dev])
        head.temperature = t
        head.meta = {"variant": name, "blocks": ",".join(blocks), "C": str(c_best)}
        test_logits = np.zeros((len(ti.image_ids), len(CLASS_NAMES)))
        if ok_test.any():
            test_logits[ok_test] = head.logits(x_test[ok_test])
        if not ok_test.all():
            assert whole_test_logits is not None
            test_logits[~ok_test] = whole_test_logits[~ok_test]
        # temperature: own T for own rows, whole-image T for fallback rows
        temp_dev = np.where(ok_dev, t, whole_T)[:, None]
        temp_test = np.where(ok_test, t, whole_T)[:, None]
        oof_scores[name] = _softmax(oof_logits / temp_dev)
        test_scores[name] = _softmax(test_logits / temp_test)
        fallback_dev[name], fallback_test[name] = ~ok_dev, ~ok_test
        head.save(heads_dir / f"{name}.npz")
        if name == "dinov2_whole":
            whole_oof_logits, whole_test_logits, whole_T = oof_logits, test_logits, t
        selection["variants"][name] = {
            "role": role,
            "blocks": list(blocks),
            "C_grid": grid,
            "C": c_best,
            "dev_oof_balanced_accuracy": bacc_best,
            "temperature": t,
            "dev_fallback_to_whole_image": int((~ok_dev).sum()),
            "test_fallback_to_whole_image": int((~ok_test).sum()),
        }
        log(
            f"classifiers: {name:22s} C={c_best:<6} dev OOF bal.acc={bacc_best:.3f} T={t:.2f} "
            f"fallback dev/test={(~ok_dev).sum()}/{(~ok_test).sum()}"
        )

    # Frozen operating points from development OOF scores.
    cov = cfg["abstention"]["target_dev_coverage"]
    for name, s in oof_scores.items():
        v = selection["variants"][name]
        v["abstention_threshold"] = float(np.quantile(s.max(1), 1 - cov))
        fpr, tpr, thr = roc_curve((y == HGS_INDEX).astype(int), s[:, HGS_INDEX])
        j = int(np.argmax(tpr - fpr))
        v["hgs_threshold"] = float(thr[j]) if np.isfinite(thr[j]) else 1.0
        v["hgs_threshold_dev_oof"] = {"sensitivity": float(tpr[j]), "specificity": float(1 - fpr[j])}

    # Primary deployed pathway: best dev OOF balanced accuracy among deployable Meta pathways.
    primary = max(DEPLOYABLE, key=lambda n: selection["variants"][n]["dev_oof_balanced_accuracy"])
    selection["primary_pathway"] = primary
    selection["fallback_pathway"] = "dinov2_whole"
    selection["primary_rule"] = (
        "Among deployable DINOv2 pathways (whole image; whole image + automatic SAM ROI), the one "
        "with higher development out-of-fold balanced accuracy. Chosen before test evaluation."
    )

    # Novelty heuristic on whole-image DINOv2 features (development only).
    x_dev = _blocks(cfg, ("dino_global",), dv.image_ids, "dev")
    scaler = StandardScaler().fit(x_dev)
    n_comp = cfg["novelty"]["pca_components"]
    pca = PCA(n_components=n_comp, random_state=seed).fit(scaler.transform(x_dev))
    z = pca.transform(scaler.transform(x_dev)) / np.sqrt(pca.explained_variance_)
    d_dev = np.sqrt((z**2).sum(1))
    thr = float(np.percentile(d_dev, cfg["novelty"]["dev_percentile"]))
    np.savez(
        heads_dir / "novelty.npz",
        mean=scaler.mean_,
        scale=np.where(scaler.scale_ == 0, 1, scaler.scale_),
        components=pca.components_,
        pca_mean=pca.mean_,
        variance=pca.explained_variance_,
        threshold=np.array(thr),
    )
    selection["novelty"] = {
        "method": f"Mahalanobis distance in {n_comp}-component PCA of development DINOv2 features",
        "threshold": thr,
        "dev_percentile": cfg["novelty"]["dev_percentile"],
        "validated": False,
    }

    dev_arrays: dict[str, Any] = {"image_ids": np.array(dv.image_ids), "labels": y}
    dev_arrays.update({f"{k}__scores": v for k, v in oof_scores.items()})
    dev_arrays.update({f"{k}__fallback": v for k, v in fallback_dev.items()})
    np.savez(out_dir / "dev_oof_scores.npz", **dev_arrays)
    test_arrays: dict[str, Any] = {"image_ids": np.array(ti.image_ids)}
    test_arrays.update({f"{k}__scores": v for k, v in test_scores.items()})
    test_arrays.update({f"{k}__fallback": v for k, v in fallback_test.items()})
    np.savez(out_dir / "test_scores_unlabelled.npz", **test_arrays)
    selection["frozen_at"] = datetime.now(UTC).isoformat()
    selection["test_labels_read"] = False
    sel_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    log(f"classifiers: primary pathway = {primary}")
    return selection
