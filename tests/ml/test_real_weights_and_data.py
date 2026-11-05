"""Tests that need real Meta weights (slow) or the local MMOTU copy (data). Skipped otherwise."""

import collections
import os

import numpy as np
import pytest

from ovascope_ml import paths

slow = pytest.mark.skipif(
    os.environ.get("OVASCOPE_RUN_SLOW") != "1", reason="set OVASCOPE_RUN_SLOW=1 to load real weights"
)
has_data = (paths.mmotu_root() / "OTU_2d" / "train_cls.txt").exists()
data = pytest.mark.skipif(not has_data, reason="local MMOTU OTU_2d copy not present")


@pytest.mark.slow
@slow
def test_dinov2_loads_offline_from_pinned_local_file():
    from ovascope_ml.models.backbones import Dinov2Extractor

    ext = Dinov2Extractor("dinov2_vits14", img_size=448)
    ok, why = ext.available()
    assert ok, why
    out = ext.extract([np.zeros((300, 400, 3), np.uint8) + 60])[0]
    assert out.patches.shape == (32, 32, 384) and out.global_features().shape == (768,)
    assert np.isfinite(out.global_features()).all()


@pytest.mark.slow
@slow
def test_sam21_box_prompt_on_synthetic_shape():
    from PIL import Image, ImageDraw

    from ovascope_ml.models.geometry import dice_iou
    from ovascope_ml.providers.sam import Sam2BoxRefiner

    img = Image.new("RGB", (640, 480), (15, 15, 15))
    ImageDraw.Draw(img).ellipse((200, 150, 420, 330), fill=(200, 200, 200))
    rgb = np.asarray(img)
    r = Sam2BoxRefiner()
    assert r.status().available
    m = r.refine(rgb, [(190, 140, 430, 340)])[0]
    yy, xx = np.mgrid[:480, :640]
    ref = ((xx - 310) / 110) ** 2 + ((yy - 240) / 90) ** 2 <= 1
    assert m.mask.shape == (480, 640) and dice_iou(m.mask, ref)[0] > 0.9


@pytest.mark.data
@data
def test_mmotu_manifest_matches_published_counts():
    from ovascope_ml.data.mmotu import build_manifest

    m = build_manifest(paths.mmotu_root(), compute_hashes=False)
    assert len(m.records) == 1469
    dev = collections.Counter(r.label_normalized for r in m.by_split("development"))
    test = collections.Counter(r.label_normalized for r in m.by_split("test"))
    # Counts from the paper's category figure (arXiv 2207.06799v3).
    assert [
        dev[c]
        for c in (
            "chocolate_cyst",
            "serous_cystadenoma",
            "teratoma",
            "theca_cell_tumor",
            "simple_cyst",
            "normal_ovary",
            "mucinous_cystadenoma",
            "high_grade_serous",
        )
    ] == [226, 153, 228, 57, 47, 180, 71, 38]
    assert [
        test[c]
        for c in (
            "chocolate_cyst",
            "serous_cystadenoma",
            "teratoma",
            "theca_cell_tumor",
            "simple_cyst",
            "normal_ovary",
            "mucinous_cystadenoma",
            "high_grade_serous",
        )
    ] == [110, 66, 108, 31, 19, 87, 33, 15]
    assert all(r.patient_id is None for r in m.records)  # never fabricated
    assert all(r.stage_if_available is None for r in m.records)
    assert m.source.patient_independent is False and m.source.external_api_transmission_authorized is False


@pytest.mark.slow
@slow
@data
def test_live_engine_reproduces_evaluated_predictions():
    """The deployed engine must give the scores the held-out evaluation measured (train/inference parity)."""
    import csv

    from ovascope_ml.inference.pipeline import AnalysisEngine

    exp = paths.experiments_dir() / "mmotu-otu2d-v1"
    pred_file = exp / "predictions_test.csv"
    if not pred_file.exists() or not (paths.bundles_dir() / "ACTIVE").exists():
        pytest.skip("no evaluated experiment / active bundle")
    eng = AnalysisEngine.from_environment()
    primary = eng.bundle.meta["primary_pathway"]
    rows = [r for r in csv.DictReader(pred_file.open(encoding="utf-8")) if r["variant"] == primary][:3]
    for r in rows:
        data = (paths.mmotu_root() / "OTU_2d" / "images" / f"{r['image_id'].removeprefix('mmotu2d-')}.JPG").read_bytes()
        res = eng.analyze(data, "parity").result
        ref = {k.removeprefix("score_"): float(v) for k, v in r.items() if k.startswith("score_")}
        assert max(abs(res.model_score_or_null[k] - ref[k]) for k in ref) < 1e-5
        assert (res.abstention_status == "abstained") == (r["abstained"] == "True")
