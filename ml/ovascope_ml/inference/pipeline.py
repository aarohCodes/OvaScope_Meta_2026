"""Analysis engine: upload bytes in, typed result + overlay artifacts out.

Honesty rules enforced here:
  * `analyze()` receives image bytes and a study id only. It has no access to
    labels, reference masks, file names or the dataset manifest.
  * Without an integrity-checked model bundle and local DINOv2 weights the engine
    is in NO_MODEL mode and returns `model_unavailable` with no scores.
  * A segmentation mask is returned only when SAM produced it. If localisation or
    SAM fails, the result says so and, when the primary pathway needs a mask,
    the independently evaluated whole-image pathway is used and labelled.
  * Any model failure yields `failed` with an error code, never a fallback label.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ovascope_ml import paths
from ovascope_ml.data.imaging import ImageInputError, ImageLimits, decode_image
from ovascope_ml.inference import render
from ovascope_ml.inference.quality import image_checks, technical_pass
from ovascope_ml.inference.result_schema import (
    AnalysisResult,
    CalibrationStatus,
    CategoryScore,
    ClassificationInfo,
    ErrorInfo,
    MissingRequirement,
    QualityCheck,
    ResultStatus,
    SegmentationInfo,
)
from ovascope_ml.models.backbones import Dinov2Extractor
from ovascope_ml.models.geometry import expand_box, mask_bbox
from ovascope_ml.models.heads import LinearHead
from ovascope_ml.models.localizer import LocalizerParams, PatchLocalizer
from ovascope_ml.providers.base import ProviderError, ProviderStatus
from ovascope_ml.providers.meta_model_api import make_multimodal_provider
from ovascope_ml.providers.sam import Sam2BoxRefiner

log = logging.getLogger("ovascope.engine")

SCORE_SEMANTICS = (
    "Temperature-scaled model scores across the eight MMOTU categories. They are not calibrated "
    "probabilities and not a personal risk estimate."
)
MASK_LIMITATION = (
    "Region delineated by SAM 2.1 from an automatic DINOv2 box prompt. It marks a candidate region "
    "(lesion, or ovary for normal-ovary images per the MMOTU convention). It is not a diagnosis."
)
PRIOR_LIMITATION = "Coarse 14-pixel patch scores from the DINOv2 localiser that produced the SAM prompt."
ATTRIBUTION_LIMITATION = (
    "Linear decomposition of the whole-image mean-patch-token term of the highest-scoring category's "
    "logit (relative to the mean over categories). Patch tokens mix information across the image, so "
    "this shows model sensitivity, not where disease is. It omits the CLS-token term and, on the ROI "
    "pathway, the ROI term."
)
_FILE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class Cancelled(Exception):
    pass


class BundleError(RuntimeError):
    pass


@dataclass
class EngineArtifact:
    kind: Literal["segmentation_mask", "localizer_prior", "attribution_map"]
    png: bytes
    width: int
    height: int
    source: str
    description: str
    limitations: str


@dataclass
class EngineOutput:
    result: AnalysisResult
    artifacts: list[EngineArtifact]


class LoadedBundle:
    def __init__(self, directory: Path) -> None:
        from ovascope_ml.training.bundle import BUNDLE_SCHEMA, file_map_hash

        self.dir = directory
        meta_path = directory / "bundle.json"
        if not meta_path.exists():
            raise BundleError(f"bundle.json missing in bundle '{directory.name}'")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("schema_version") != BUNDLE_SCHEMA:
            raise BundleError("unsupported bundle schema version")
        files = self.meta.get("files", {})
        for name, sha in files.items():
            if not _FILE_RE.match(name):
                raise BundleError("bundle lists an invalid file name")
            p = directory / name
            if not p.exists():
                raise BundleError(f"bundle file missing: {name}")
            if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
                raise BundleError(f"bundle integrity check failed for {name}")
        if file_map_hash(files) != self.meta.get("checkpoint_sha256"):
            raise BundleError("bundle checkpoint hash does not match its file list")
        self.heads = {k: LinearHead.load(directory / v["head"]) for k, v in self.meta["pathways"].items()}
        seg = self.meta["segmentation"]
        self.localizer = PatchLocalizer(
            LinearHead.load(directory / seg["localizer_head"]),
            LocalizerParams.load(directory / seg["localizer_params"]),
        )
        with np.load(directory / self.meta["novelty"]["file"], allow_pickle=False) as z:
            self.novelty = {k: z[k] for k in z.files}

    @property
    def checkpoint_sha256(self) -> str:
        return str(self.meta["checkpoint_sha256"])


def resolve_active_bundle(bundles_dir: Path) -> tuple[LoadedBundle | None, str | None]:
    explicit = os.environ.get("OVASCOPE_BUNDLE_ID")
    active_file = bundles_dir / "ACTIVE"
    bundle_id = explicit or (active_file.read_text(encoding="utf-8").strip() if active_file.exists() else None)
    if not bundle_id:
        return None, "No trained model bundle is active (no artifacts/bundles/ACTIVE and no OVASCOPE_BUNDLE_ID)."
    if not _FILE_RE.match(bundle_id):
        return None, "Configured bundle id is malformed."
    d = bundles_dir / bundle_id
    if not d.is_dir():
        return None, f"Configured bundle '{bundle_id}' does not exist."
    try:
        return LoadedBundle(d), None
    except BundleError as exc:
        return None, f"Bundle '{bundle_id}' rejected: {exc}"
    except Exception as exc:  # malformed files
        return None, f"Bundle '{bundle_id}' could not be loaded ({type(exc).__name__})."


def _timed(fn: Callable[[], object], seconds: float) -> object:
    ex = cf.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=seconds)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


class AnalysisEngine:
    def __init__(
        self,
        bundle: LoadedBundle | None,
        bundle_error: str | None,
        force_no_model: bool = False,
        segmentation_timeout_s: float = 90.0,
        limits: ImageLimits | None = None,
        sam: Sam2BoxRefiner | None = None,
    ) -> None:
        self.bundle = bundle
        self.bundle_error = bundle_error
        self.force_no_model = force_no_model
        self.segmentation_timeout_s = segmentation_timeout_s
        self.limits = limits or ImageLimits()
        key = bundle.meta["backbone"]["key"] if bundle else "dinov2_vits14"
        size = bundle.meta["backbone"]["img_size"] if bundle else 448
        self.dino = Dinov2Extractor(key, img_size=size)
        self.roi_dino = Dinov2Extractor(key, img_size=bundle.meta["roi"]["img_size"] if bundle else 224)
        sam_key = bundle.meta["segmentation"]["sam_key"] if bundle else "sam2.1_hiera_tiny"
        multimask = bundle.meta["segmentation"]["multimask_output"] if bundle else False
        self.sam = sam or Sam2BoxRefiner(sam_key, multimask_output=multimask)
        self.multimodal = make_multimodal_provider()
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> AnalysisEngine:
        mode = os.environ.get("OVASCOPE_INFERENCE_MODE", "auto").lower()
        if mode not in ("auto", "no_model"):
            mode = "auto"
        bundle, err = resolve_active_bundle(paths.bundles_dir())
        if mode == "no_model":
            err = "Inference disabled by configuration (OVASCOPE_INFERENCE_MODE=no_model)."
        return cls(
            bundle,
            err,
            force_no_model=mode == "no_model",
            segmentation_timeout_s=float(os.environ.get("OVASCOPE_SEGMENTATION_TIMEOUT_S", "90")),
            limits=ImageLimits(max_bytes=int(os.environ.get("OVASCOPE_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))),
        )

    # ---------------------------------------------------------------- status
    def _classifier_status(self) -> ProviderStatus:
        if self.bundle is None or self.force_no_model:
            return ProviderStatus(
                kind="classifier",
                provider_id="ovascope-bundle",
                available=False,
                reason_unavailable=self.bundle_error,
                action_required="Train and export a bundle: make data-audit train evaluate bundle",
            )
        b = self.bundle.meta
        return ProviderStatus(
            kind="classifier",
            provider_id="ovascope-bundle",
            available=True,
            model_id=b["model_id"],
            model_version=b["model_version"],
            details={
                "bundle_id": b["bundle_id"],
                "checkpoint_sha256": b["checkpoint_sha256"],
                "target": b["classification_target"],
                "primary_pathway": b["primary_pathway"],
            },
        )

    def _representation_status(self) -> ProviderStatus:
        ok, why = self.dino.available()
        return ProviderStatus(
            kind="representation",
            provider_id="dinov2",
            available=ok,
            model_id=self.dino.spec.repo_id,
            model_version=self.dino.spec.revision,
            license=self.dino.spec.license,
            meta_model=True,
            reason_unavailable=why,
            action_required=None if ok else "ovascope-ml fetch-models",
        )

    def _segmentation_status(self) -> ProviderStatus:
        sam = self.sam.status()
        rep_ok, _ = self.dino.available()
        if not sam.available:
            return sam
        if self.bundle is None or self.force_no_model or not rep_ok:
            return ProviderStatus(
                kind="segmentation",
                provider_id="dinov2-localizer+sam2.1",
                available=False,
                model_id=sam.model_id,
                meta_model=True,
                license=sam.license,
                reason_unavailable="Automatic prompts need the trained DINOv2 localiser from a model bundle.",
                action_required="Export a model bundle (make bundle)",
            )
        return ProviderStatus(
            kind="segmentation",
            provider_id="dinov2-localizer+sam2.1",
            available=True,
            model_id=sam.model_id,
            model_version=sam.model_version,
            meta_model=True,
            license=sam.license,
            details={"prompt": "automatic box from DINOv2 patch localiser"},
        )

    def statuses(self) -> list[ProviderStatus]:
        return [
            self._representation_status(),
            self._segmentation_status(),
            self._classifier_status(),
            self.multimodal.status(),
        ]

    def mode(self) -> str:
        rep_ok, _ = self.dino.available()
        return "RESEARCH_INFERENCE" if (self.bundle is not None and not self.force_no_model and rep_ok) else "NO_MODEL"

    def missing_requirements(self) -> list[MissingRequirement]:
        out = []
        for st in self.statuses():
            if not st.available and st.kind != "multimodal":
                out.append(
                    MissingRequirement(
                        component=st.kind,
                        requirement=st.reason_unavailable or "unavailable",
                        action=st.action_required or "see docs/README",
                    )
                )
        return out

    # ---------------------------------------------------------------- analysis
    def analyze(
        self,
        data: bytes,
        study_id: str,
        cancel_check: Callable[[], bool] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> EngineOutput:
        with self._lock:
            return self._analyze(data, study_id, cancel_check or (lambda: False), on_stage or (lambda s: None))

    def _base(self, study_id: str, sha: str) -> AnalysisResult:
        r = AnalysisResult(study_id=study_id, status=ResultStatus.failed, input_sha256=sha)
        if self.bundle is not None and not self.force_no_model:
            b = self.bundle.meta
            r.model_id, r.model_version, r.checkpoint_sha256 = b["model_id"], b["model_version"], b["checkpoint_sha256"]
            r.classification_target = b["classification_target"]
            r.class_mapping = b["class_mapping"]
            r.evaluation_reference = b["evaluation_reference"]
        return r

    def _analyze(
        self, data: bytes, study_id: str, cancelled: Callable[[], bool], stage: Callable[[str], None]
    ) -> EngineOutput:
        times: dict[str, float] = {}
        t_all = time.perf_counter()

        def mark(name: str, t0: float) -> None:
            times[name] = round((time.perf_counter() - t0) * 1000, 1)

        def check_cancel() -> None:
            if cancelled():
                raise Cancelled()

        sha = hashlib.sha256(data).hexdigest()
        result = self._base(study_id, sha)
        stage("validating")
        t0 = time.perf_counter()
        try:
            img = decode_image(data, self.limits)
        except ImageInputError as exc:
            result.status = ResultStatus.unsupported_input
            result.error = ErrorInfo(code=exc.code.value, message=exc.message)
            result.quality_checks = [
                QualityCheck(id="decode", category="technical", status="fail", message=exc.message)
            ]
            result.model_id = result.model_version = result.checkpoint_sha256 = None
            mark("validation", t0)
            result.processing_times = times
            return EngineOutput(result, [])
        rgb = img.rgb
        result.quality_checks = technical_pass(img.format, img.width, img.height) + image_checks(rgb)
        mark("validation", t0)

        if self.mode() != "RESEARCH_INFERENCE":
            result.status = ResultStatus.model_unavailable
            result.missing_requirements = self.missing_requirements()
            result.segmentation = SegmentationInfo(
                status="not_run", message="No model bundle: automatic segmentation not run."
            )
            result.processing_times = {**times, "total": round((time.perf_counter() - t_all) * 1000, 1)}
            return EngineOutput(result, [])

        assert self.bundle is not None
        b = self.bundle
        artifacts: list[EngineArtifact] = []
        try:
            check_cancel()
            stage("representation")
            t0 = time.perf_counter()
            dino_out = self.dino.extract([rgb])[0]
            mark("representation", t0)

            check_cancel()
            stage("segmentation")
            t0 = time.perf_counter()
            seg, mask, prior = self._segment(rgb, dino_out)
            mark("segmentation", t0)
            result.segmentation = seg
            result.segmentation_provider = f"{seg.provider_id} ({seg.model_id})" if seg.model_id else seg.provider_id
            if prior is not None:
                artifacts.append(
                    EngineArtifact(
                        "localizer_prior",
                        render.heat_png(prior),
                        rgb.shape[1],
                        rgb.shape[0],
                        f"DINOv2 patch localiser ({self.dino.model_id})",
                        "Localisation prior that produced the SAM box prompt",
                        PRIOR_LIMITATION,
                    )
                )
            if mask is not None:
                artifacts.append(
                    EngineArtifact(
                        "segmentation_mask",
                        render.mask_png(mask),
                        rgb.shape[1],
                        rgb.shape[0],
                        f"SAM 2.1 ({self.sam.model_id}), automatic box prompt",
                        "Segmentation mask: candidate region delineated by SAM",
                        MASK_LIMITATION,
                    )
                )

            check_cancel()
            stage("classification")
            t0 = time.perf_counter()
            primary = b.meta["primary_pathway"]
            whole = dino_out.global_features()[None, :]
            pathway: Literal["dinov2_roi_sam", "dinov2_whole"]
            if primary == "dinov2_roi_sam" and mask is not None:
                box = mask_bbox(mask)
                assert box is not None
                x1, y1, x2, y2 = expand_box(box, b.meta["roi"]["margin"], rgb.shape[1], rgb.shape[0])
                roi = self.roi_dino.extract([rgb[y1:y2, x1:x2]])[0].global_features()[None, :]
                pathway, feats = "dinov2_roi_sam", np.concatenate([whole, roi], axis=1)
                reason = "Primary pathway: whole image + automatic SAM region of interest."
            elif primary == "dinov2_roi_sam":
                pathway, feats = "dinov2_whole", whole
                reason = (
                    f"Fallback: segmentation {seg.status.replace('_', ' ')}; the independently evaluated "
                    "whole-image pathway produced this result."
                )
                result.warnings.append(
                    "Whole-image fallback used because automatic segmentation did not produce a mask."
                )
            else:
                pathway, feats = "dinov2_whole", whole
                reason = "Primary pathway selected on development data: whole image (ROI did not improve development performance)."
            head = b.heads[pathway]
            scores = head.scores(feats)[0]
            mark("classification", t0)
            mapping = {c["label"]: c["display"] for c in b.meta["class_mapping"]}
            ranked = sorted(zip(head.classes, scores.tolist(), strict=True), key=lambda t: -t[1])
            top_label, top_score = ranked[0]
            result.classification = ClassificationInfo(
                pathway=pathway,
                pathway_reason=reason,
                scores=[CategoryScore(label=k, display=mapping[k], score=round(v, 6)) for k, v in ranked],
                highest_scoring_label=top_label,
                highest_scoring_display=mapping[top_label],
                score_semantics=SCORE_SEMANTICS,
            )
            result.model_score_or_null = {k: round(v, 6) for k, v in zip(head.classes, scores.tolist(), strict=True)}
            result.calibration_status = CalibrationStatus.temperature_scaled_not_prevalence_calibrated

            thr = float(b.meta["abstention"]["thresholds"][pathway])
            if top_score < thr:
                result.abstention_status = "abstained"
                result.abstention_reason_or_null = (
                    f"Highest category score {top_score:.2f} is below the abstention threshold {thr:.2f} "
                    "frozen on development data (target 90% coverage)."
                )
            else:
                result.abstention_status = "not_abstained"

            nov = self._novelty(whole[0])
            result.quality_checks.append(nov)

            check_cancel()
            stage("attribution")
            t0 = time.perf_counter()
            attr = self._attribution(head, dino_out, top_label)
            if attr is not None:
                artifacts.append(
                    EngineArtifact(
                        "attribution_map",
                        render.diverging_png(attr),
                        rgb.shape[1],
                        rgb.shape[0],
                        f"Linear patch decomposition of {pathway} head",
                        "Attribution map: method-dependent model sensitivity",
                        ATTRIBUTION_LIMITATION,
                    )
                )
            mark("attribution", t0)
        except Cancelled:
            raise
        except ProviderError as exc:
            log.warning("provider failure code=%s", exc.code)
            return self._failed(result, exc.code, str(exc), times, t_all, retryable=exc.code == "PROVIDER_TIMEOUT")
        except Exception as exc:
            log.exception("inference failure type=%s", type(exc).__name__)
            return self._failed(
                result, "INFERENCE_FAILED", f"Model inference failed ({type(exc).__name__}).", times, t_all
            )

        warn = any(q.status == "warn" for q in result.quality_checks) or bool(result.warnings)
        if result.abstention_status == "abstained":
            result.status = ResultStatus.abstained
        else:
            result.status = ResultStatus.completed_with_warnings if warn else ResultStatus.completed
        result.processing_times = {**times, "total": round((time.perf_counter() - t_all) * 1000, 1)}
        return EngineOutput(result, artifacts)

    def _failed(
        self, result: AnalysisResult, code: str, msg: str, times, t_all, retryable: bool = False
    ) -> EngineOutput:
        result.status = ResultStatus.failed
        result.error = ErrorInfo(code=code, message=msg, retryable=retryable)
        result.classification = None
        result.model_score_or_null = None
        result.abstention_status = "not_applicable"
        result.calibration_status = CalibrationStatus.not_applicable
        result.processing_times = {**times, "total": round((time.perf_counter() - t_all) * 1000, 1)}
        return EngineOutput(result, [])

    def _segment(self, rgb: np.ndarray, dino_out) -> tuple[SegmentationInfo, np.ndarray | None, np.ndarray | None]:
        assert self.bundle is not None
        loc = self.bundle.localizer.localize(dino_out.patches.astype(np.float32), dino_out.letterbox)
        prior = dino_out.letterbox.model_map_to_source(loc.prob_grid)
        base: dict[str, Any] = {
            "localizer_model_id": f"dinov2 patch localiser ({self.bundle.meta['bundle_id']})",
            "localizer_max_score": round(loc.max_score, 4),
        }
        if loc.status != "candidate_found" or loc.box is None:
            return (
                SegmentationInfo(
                    status="no_candidate_region",
                    provider_id="dinov2-localizer",
                    model_id=None,
                    message="The localiser found no region above its development-selected threshold; "
                    "SAM was not prompted and no mask is shown.",
                    **base,
                ),
                None,
                prior,
            )
        sam_status = self.sam.status()
        if not sam_status.available:
            return (
                SegmentationInfo(
                    status="provider_unavailable",
                    provider_id=self.sam.provider_id,
                    model_id=self.sam.spec.repo_id,
                    prompt_source="dinov2-localizer",
                    prompt_box_xyxy=list(loc.box),
                    message=sam_status.reason_unavailable,
                    **base,
                ),
                None,
                prior,
            )
        try:
            out = _timed(lambda: self.sam.refine(rgb, [loc.box]), self.segmentation_timeout_s)  # type: ignore[list-item]
        except cf.TimeoutError:
            return (
                SegmentationInfo(
                    status="timeout",
                    provider_id=self.sam.provider_id,
                    model_id=self.sam.model_id,
                    prompt_source="dinov2-localizer",
                    prompt_box_xyxy=list(loc.box),
                    message=f"SAM exceeded {self.segmentation_timeout_s:.0f}s.",
                    **base,
                ),
                None,
                prior,
            )
        except ProviderError as exc:
            return (
                SegmentationInfo(
                    status="provider_failed",
                    provider_id=self.sam.provider_id,
                    model_id=self.sam.model_id,
                    prompt_source="dinov2-localizer",
                    prompt_box_xyxy=list(loc.box),
                    message=str(exc),
                    **base,
                ),
                None,
                prior,
            )
        sm = out[0]  # type: ignore[index]
        if not sm.mask.any():
            return (
                SegmentationInfo(
                    status="provider_failed",
                    provider_id=self.sam.provider_id,
                    model_id=self.sam.model_id,
                    prompt_source="dinov2-localizer",
                    prompt_box_xyxy=list(loc.box),
                    message="SAM returned an empty mask.",
                    **base,
                ),
                None,
                prior,
            )
        return (
            SegmentationInfo(
                status="succeeded",
                provider_id=self.sam.provider_id,
                model_id=self.sam.model_id,
                prompt_source="dinov2-localizer (automatic box)",
                prompt_box_xyxy=list(loc.box),
                sam_predicted_iou=round(sm.predicted_iou, 4),
                mask_area_fraction=round(float(sm.mask.mean()), 4),
                **base,
            ),
            sm.mask,
            prior,
        )

    def _novelty(self, whole: np.ndarray) -> QualityCheck:
        assert self.bundle is not None
        n = self.bundle.novelty
        z = ((whole - n["mean"]) / n["scale"] - n["pca_mean"]) @ n["components"].T / np.sqrt(n["variance"])
        d = float(np.sqrt((z**2).sum()))
        thr = float(n["threshold"])
        atypical = d > thr
        return QualityCheck(
            id="feature_novelty",
            category="distribution_shift",
            status="warn" if atypical else "pass",
            value=round(d, 3),
            message=(
                f"Input is atypical relative to development images (distance {d:.1f} > {thr:.1f}). "
                if atypical
                else f"Feature distance {d:.1f} is within the development 99th percentile ({thr:.1f}). "
            )
            + "Heuristic; not validated out-of-distribution detection.",
        )

    def _attribution(self, head: LinearHead, dino_out, top_label: str) -> np.ndarray | None:
        d = dino_out.patches.shape[-1]
        k = head.classes.index(top_label)
        sl = slice(d, 2 * d)  # whole-image mean-patch-token block in both heads
        w = (head.coef[k, sl] - head.coef[:, sl].mean(axis=0)) / head.scale[sl]
        g = dino_out.patches.shape[0]
        tokens = dino_out.patches.astype(np.float64).reshape(g * g, d)
        contrib = ((tokens - head.mean[sl]) @ w / (g * g)).reshape(g, g)
        return dino_out.letterbox.model_map_to_source(contrib)
