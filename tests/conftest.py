"""Shared fixtures. All images here are synthetic, non-medical geometric shapes
used for file-handling and rendering tests only; they are never presented as
medical data and never used for evaluation."""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ovascope_ml.models.geometry import Letterbox
from ovascope_ml.models.heads import LinearHead
from ovascope_ml.models.localizer import LocalizerParams
from ovascope_ml.training.bundle import BUNDLE_SCHEMA, file_map_hash
from ovascope_ml.training.common import CLASS_NAMES

D = 384  # DINOv2 ViT-S/14 embedding width
G = 32  # patch grid at 448 px


def geometric_png(w: int = 400, h: int = 300, shape: str = "ellipse", seed: int = 0, fmt: str = "PNG") -> bytes:
    rng = np.random.default_rng(seed)
    base = (rng.random((h, w)) * 40).astype(np.uint8)
    im = Image.fromarray(np.stack([base] * 3, -1))
    d = ImageDraw.Draw(im)
    if shape == "ellipse":
        d.ellipse((w * 0.3, h * 0.3, w * 0.7, h * 0.7), fill=(200, 200, 200))
    elif shape == "rect":
        d.rectangle((w * 0.1, h * 0.2, w * 0.5, h * 0.8), fill=(150, 150, 150))
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()


def slow_enabled() -> bool:
    return os.environ.get("OVASCOPE_RUN_SLOW") == "1"


# ---------------------------------------------------------------- synthetic bundle
def write_synthetic_bundle(root: Path, primary: str = "dinov2_roi_sam", abstain_threshold: float = 0.0) -> Path:
    """A structurally valid bundle with random heads. For engine plumbing tests ONLY."""
    rng = np.random.default_rng(0)
    d = root / "synthetic-test-bundle"
    d.mkdir(parents=True)
    k = len(CLASS_NAMES)

    def head(dim: int) -> LinearHead:
        return LinearHead(
            mean=np.zeros(dim),
            scale=np.ones(dim),
            coef=rng.normal(size=(k, dim)) * 0.01,
            intercept=np.zeros(k),
            classes=list(CLASS_NAMES),
            temperature=1.0,
        )

    head(2 * D).save(d / "head_dinov2_whole.npz")
    head(4 * D).save(d / "head_dinov2_roi_sam.npz")
    loc = LinearHead(
        mean=np.zeros(D),
        scale=np.ones(D),
        coef=np.zeros((1, D)),
        intercept=np.array([3.0]),
        classes=["background", "target_region"],
        kind="sigmoid",
    )
    loc.save(d / "localizer_head.npz")
    LocalizerParams(threshold=0.5, smooth_sigma=0.0, min_component_patches=1).save(d / "localizer_params.json")
    np.savez(
        d / "novelty.npz",
        mean=np.zeros(2 * D),
        scale=np.ones(2 * D),
        components=np.eye(4, 2 * D),
        pca_mean=np.zeros(2 * D),
        variance=np.ones(4),
        threshold=np.array(1e9),
    )
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}
    meta = {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_id": d.name,
        "model_id": "synthetic-test-model",
        "model_version": "0.0.0-test",
        "created_at": "2026-11-04T00:00:00Z",
        "classification_target": "mmotu8",
        "class_mapping": [{"index": i, "label": c, "display": c} for i, c in enumerate(CLASS_NAMES)],
        "pathways": {
            "dinov2_roi_sam": {"head": "head_dinov2_roi_sam.npz"},
            "dinov2_whole": {"head": "head_dinov2_whole.npz"},
        },
        "primary_pathway": primary,
        "fallback_pathway": "dinov2_whole",
        "backbone": {"key": "dinov2_vits14", "img_size": 448},
        "roi": {"img_size": 224, "margin": 0.15},
        "segmentation": {
            "localizer_head": "localizer_head.npz",
            "localizer_params": "localizer_params.json",
            "sam_key": "sam2.1_hiera_tiny",
            "multimask_output": False,
        },
        "calibration": {"status": "temperature_scaled_not_prevalence_calibrated"},
        "abstention": {"thresholds": {"dinov2_whole": abstain_threshold, "dinov2_roi_sam": abstain_threshold}},
        "novelty": {"file": "novelty.npz"},
        "evaluation_reference": {
            "experiment_id": "synthetic",
            "metrics_sha256": "0" * 64,
            "evaluation_unit": "image",
            "patient_independent": False,
            "external_validation": "External validation not performed.",
        },
        "files": files,
    }
    meta["checkpoint_sha256"] = file_map_hash(files)
    (d / "bundle.json").write_text(json.dumps(meta), encoding="utf-8")
    (root / "ACTIVE").write_text(d.name, encoding="utf-8")
    return d


class FakeDino:
    """Stands in for DINOv2 so engine plumbing can be tested without loading weights."""

    def __init__(self, img_size: int = 448, available: bool = True) -> None:
        self.img_size = img_size
        self._available = available
        self.model_id = "fake-dinov2"
        self.spec = type("S", (), {"repo_id": "fake/dinov2", "revision": "0", "license": "test"})()

    def available(self):
        return (True, None) if self._available else (False, "fake weights missing")

    def extract(self, images):
        from ovascope_ml.models.backbones import DinoOutput

        outs = []
        g = self.img_size // 14
        for im in images:
            lb = Letterbox(im.shape[1], im.shape[0], self.img_size)
            patches = np.zeros((g, g, D), dtype=np.float16)
            outs.append(
                DinoOutput(
                    cls=np.zeros(D, np.float32), patch_mean=np.zeros(D, np.float32), patches=patches, letterbox=lb
                )
            )
        return outs


class FakeSam:
    provider_id = "fake-sam"
    model_id = "fake/sam"

    def __init__(self, mode: str = "ok", delay: float = 0.0) -> None:
        self.mode, self.delay = mode, delay
        self.spec = type("S", (), {"repo_id": "fake/sam"})()

    def status(self):
        from ovascope_ml.providers.base import ProviderStatus

        return ProviderStatus(
            kind="segmentation",
            provider_id=self.provider_id,
            available=self.mode != "unavailable",
            model_id=self.model_id,
            reason_unavailable="fake unavailable" if self.mode == "unavailable" else None,
        )

    def refine(self, rgb, boxes, embedding=None):
        from ovascope_ml.providers.base import ProviderError
        from ovascope_ml.providers.sam import SamMask

        if self.delay:
            time.sleep(self.delay)
        if self.mode == "fail":
            raise ProviderError("SEGMENTATION_FAILED", "fake failure")
        out = []
        for x1, y1, x2, y2 in boxes:
            m = np.zeros(rgb.shape[:2], bool)
            m[y1:y2, x1:x2] = True
            out.append(SamMask(mask=m, predicted_iou=0.9))
        return out


@pytest.fixture
def synthetic_engine(tmp_path, monkeypatch):
    """Factory: engine wired to a synthetic bundle with fake DINOv2/SAM."""
    from ovascope_ml.inference.pipeline import AnalysisEngine, resolve_active_bundle

    def make(
        primary: str = "dinov2_roi_sam",
        sam_mode: str = "ok",
        sam_delay: float = 0.0,
        abstain: float = 0.0,
        timeout: float = 5.0,
    ):
        root = tmp_path / f"bundles-{primary}-{sam_mode}-{abstain}"
        write_synthetic_bundle(root, primary=primary, abstain_threshold=abstain)
        bundle, err = resolve_active_bundle(root)
        assert bundle is not None, err
        eng = AnalysisEngine(bundle, None, sam=FakeSam(sam_mode, sam_delay), segmentation_timeout_s=timeout)  # type: ignore[arg-type]
        eng.dino = FakeDino(448)  # type: ignore[assignment]
        eng.roi_dino = FakeDino(224)  # type: ignore[assignment]
        return eng

    return make


@pytest.fixture
def no_model_engine(tmp_path):
    from ovascope_ml.inference.pipeline import AnalysisEngine, resolve_active_bundle

    bundle, err = resolve_active_bundle(tmp_path / "empty-bundles")
    eng = AnalysisEngine(bundle, err)
    eng.dino = FakeDino(448)  # type: ignore[assignment]
    return eng


@pytest.fixture
def api_factory(tmp_path):
    """Build an API app on temp storage with an injected engine (inline worker on)."""
    from fastapi.testclient import TestClient

    from ovascope_api.main import create_app
    from ovascope_api.settings import Settings

    def make(engine, **overrides):
        opts = {"inline_worker": True, **overrides}
        s = Settings(
            database_url=f"sqlite:///{(tmp_path / 'api.db').as_posix()}", artifact_dir=str(tmp_path / "store"), **opts
        )
        app = create_app(s, engine=engine)
        return TestClient(app), app

    return make


def wait_job(client, job_id: str, timeout: float = 120.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/v1/jobs/{job_id}").json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish: {j}")
