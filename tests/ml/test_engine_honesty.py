"""The engine must never fabricate output: unavailable, failed and fallback states are explicit."""

import ast
import inspect
import json
from pathlib import Path

import pytest

from conftest import geometric_png, write_synthetic_bundle
from ovascope_ml.inference.pipeline import AnalysisEngine, resolve_active_bundle
from ovascope_ml.inference.result_schema import ResultStatus

INFERENCE_DIR = Path(__file__).resolve().parents[2] / "ml" / "ovascope_ml" / "inference"
FORBIDDEN = (
    "ovascope_ml.evaluation",
    "ovascope_ml.data.manifest",
    "ovascope_ml.data.mmotu",
    "ovascope_ml.data.audit",
    "ovascope_ml.training.common",
    "ovascope_ml.data.label_policy",
)


def test_inference_code_cannot_reach_labels_or_reference_masks():
    for f in INFERENCE_DIR.glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                assert not m.startswith(FORBIDDEN), f"{f.name} imports {m}"
        assert "_cls.txt" not in f.read_text(encoding="utf-8")


def test_analyze_signature_accepts_only_bytes_and_study_id():
    params = list(inspect.signature(AnalysisEngine.analyze).parameters)
    assert params == ["self", "data", "study_id", "cancel_check", "on_stage"]


def test_missing_checkpoint_gives_model_unavailable_without_scores(no_model_engine):
    out = no_model_engine.analyze(geometric_png(), "s1")
    r = out.result
    assert r.status == ResultStatus.model_unavailable
    assert r.model_score_or_null is None and r.classification is None
    assert r.segmentation_artifact_or_null is None and out.artifacts == []
    assert r.abstention_status == "not_applicable" and r.calibration_status.value == "not_applicable"
    assert any(m.component == "classifier" for m in r.missing_requirements)
    assert no_model_engine.mode() == "NO_MODEL"


def test_unsupported_input_is_typed(no_model_engine):
    r = no_model_engine.analyze(b"GIF89a....", "s1").result
    assert r.status == ResultStatus.unsupported_input and r.error.code == "UNSUPPORTED_FORMAT"
    assert r.model_score_or_null is None


def test_tampered_bundle_is_rejected(tmp_path):
    d = write_synthetic_bundle(tmp_path / "b")
    (d / "head_dinov2_whole.npz").write_bytes(b"tampered")
    bundle, err = resolve_active_bundle(tmp_path / "b")
    assert bundle is None and "integrity" in err
    meta = json.loads((d / "bundle.json").read_text())
    meta["files"]["../../evil"] = "0" * 64
    (d / "bundle.json").write_text(json.dumps(meta))
    assert resolve_active_bundle(tmp_path / "b")[0] is None


def test_missing_dinov2_weights_forces_no_model(synthetic_engine):
    from conftest import FakeDino

    eng = synthetic_engine()
    eng.dino = FakeDino(available=False)
    assert eng.mode() == "NO_MODEL"
    assert eng.analyze(geometric_png(), "s").result.status == ResultStatus.model_unavailable


def test_research_inference_happy_path_labels_every_source(synthetic_engine):
    out = synthetic_engine().analyze(geometric_png(), "s1")
    r = out.result
    assert r.status in (ResultStatus.completed, ResultStatus.completed_with_warnings)
    assert r.classification.pathway == "dinov2_roi_sam"
    assert r.segmentation.status == "succeeded" and r.segmentation.automatic
    assert "dinov2-localizer" in r.segmentation.prompt_source
    assert r.checkpoint_sha256 and r.model_id == "synthetic-test-model"
    assert "not calibrated probabilities" in r.classification.score_semantics
    kinds = {a.kind for a in out.artifacts}
    assert {"segmentation_mask", "attribution_map", "localizer_prior"} <= kinds
    assert all((a.width, a.height) == (400, 300) for a in out.artifacts)
    assert abs(sum(r.model_score_or_null.values()) - 1) < 1e-6


@pytest.mark.parametrize("sam_mode,expected", [("fail", "provider_failed"), ("unavailable", "provider_unavailable")])
def test_segmentation_failure_never_yields_a_mask_and_fallback_is_labelled(synthetic_engine, sam_mode, expected):
    out = synthetic_engine(sam_mode=sam_mode).analyze(geometric_png(), "s1")
    r = out.result
    assert r.segmentation.status == expected
    assert "segmentation_mask" not in {a.kind for a in out.artifacts}
    assert r.classification.pathway == "dinov2_whole"
    assert r.classification.pathway_reason.startswith("Fallback")
    assert r.status == ResultStatus.completed_with_warnings
    assert any("fallback" in w.lower() for w in r.warnings)


def test_provider_timeout_is_explicit(synthetic_engine):
    out = synthetic_engine(sam_delay=5.0, timeout=0.5).analyze(geometric_png(), "s1")
    assert out.result.segmentation.status == "timeout"
    assert "segmentation_mask" not in {a.kind for a in out.artifacts}


def test_abstention_uses_frozen_threshold(synthetic_engine):
    r = synthetic_engine(abstain=0.99).analyze(geometric_png(), "s1").result
    assert r.status == ResultStatus.abstained and r.abstention_status == "abstained"
    assert "below the abstention threshold" in r.abstention_reason_or_null


def test_model_failure_returns_failed_not_a_fallback_diagnosis(synthetic_engine):
    eng = synthetic_engine()

    class Boom:
        def available(self):
            return True, None

        def extract(self, images):
            raise RuntimeError("weights corrupted")

    eng.dino = Boom()
    r = eng.analyze(geometric_png(), "s1").result
    assert r.status == ResultStatus.failed and r.error.code == "INFERENCE_FAILED"
    assert r.model_score_or_null is None and r.classification is None


def test_cancellation_between_stages(synthetic_engine):
    from ovascope_ml.inference.pipeline import Cancelled

    with pytest.raises(Cancelled):
        synthetic_engine().analyze(geometric_png(), "s1", cancel_check=lambda: True)
