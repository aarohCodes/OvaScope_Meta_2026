"""SAM 2.1 (Meta) box-prompted mask refinement via transformers.Sam2Model.

API contract verified against transformers 4.56.2 source:
  * processor(images=..., input_boxes=[[[x1, y1, x2, y2]]]) with ORIGINAL pixel coords;
    the processor rescales boxes to the 1024x1024 model input (no padding).
  * model.get_image_embeddings(pixel_values) -> list of multi-level features, reusable
    across prompts via model(image_embeddings=..., input_boxes=...).
  * processor.post_process_masks(pred_masks, original_sizes) bilinearly resizes logits to
    the original size and thresholds at 0. (The library docstring example passes
    `reshaped_input_sizes` positionally into `mask_threshold`; we do not.)
Weights are loaded with local_files_only=True; nothing is downloaded at inference time.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from ovascope_ml.models.backbones import cached_weight_path
from ovascope_ml.models.registry import get
from ovascope_ml.providers.base import ProviderError, ProviderStatus, ProviderUnavailable


@dataclass
class SamMask:
    mask: np.ndarray  # (H, W) bool
    predicted_iou: float  # SAM's own IoU estimate; NOT a validated quality measure


class Sam2BoxRefiner:
    provider_id = "sam2.1-box-refiner"

    def __init__(self, key: str = "sam2.1_hiera_tiny", multimask_output: bool = False) -> None:
        self.spec = get(key)
        self.multimask_output = multimask_output
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return f"{self.spec.repo_id}@{self.spec.revision[:12]}"

    def status(self) -> ProviderStatus:
        try:
            import transformers

            has_cls = hasattr(transformers, "Sam2Model")
        except ImportError:
            has_cls = False
        missing = [f for f in self.spec.files if cached_weight_path(self.spec, f) is None]
        if not has_cls:
            return ProviderStatus(
                kind="segmentation",
                provider_id=self.provider_id,
                available=False,
                model_id=self.spec.repo_id,
                license=self.spec.license,
                meta_model=True,
                reason_unavailable="transformers with Sam2Model is not installed",
                action_required="pip install -e '.[ml]'",
            )
        if missing:
            return ProviderStatus(
                kind="segmentation",
                provider_id=self.provider_id,
                available=False,
                model_id=self.spec.repo_id,
                license=self.spec.license,
                meta_model=True,
                reason_unavailable=f"SAM 2.1 weights not in local cache (missing: {', '.join(missing)})",
                action_required="ovascope-ml fetch-models",
            )
        return ProviderStatus(
            kind="segmentation",
            provider_id=self.provider_id,
            available=True,
            model_id=self.spec.repo_id,
            model_version=self.spec.revision,
            license=self.spec.license,
            meta_model=True,
            execution="local",
            details={"prompt": "box", "multimask_output": self.multimask_output},
        )

    def _load(self):
        with self._lock:
            if self._model is None:
                st = self.status()
                if not st.available:
                    raise ProviderUnavailable(st.reason_unavailable or "SAM unavailable")
                import transformers
                from transformers import Sam2Model, Sam2Processor

                # The checkpoint config declares model_type "sam2_video"; loading it into the
                # image model is supported but logs a warning. Silence only that logger.
                transformers.logging.set_verbosity_error()
                self._model = Sam2Model.from_pretrained(
                    self.spec.repo_id, revision=self.spec.revision, local_files_only=True
                ).eval()
                self._processor = Sam2Processor.from_pretrained(
                    self.spec.repo_id, revision=self.spec.revision, local_files_only=True
                )
        return self._model, self._processor

    def embed(self, rgb: np.ndarray):
        import torch
        from PIL import Image

        model, proc = self._load()
        inputs = proc(images=Image.fromarray(rgb), return_tensors="pt")
        with torch.inference_mode():
            emb = model.get_image_embeddings(inputs["pixel_values"])
        return emb, inputs["original_sizes"]

    def refine(self, rgb: np.ndarray, boxes: list[tuple[int, int, int, int]], embedding=None) -> list[SamMask]:
        """One mask per box prompt (source pixel coordinates)."""
        import torch

        if not boxes:
            return []
        model, proc = self._load()
        try:
            emb, original_sizes = embedding if embedding is not None else self.embed(rgb)
            prompt = proc(
                original_sizes=original_sizes,
                input_boxes=[[list(map(float, b)) for b in boxes]],
                return_tensors="pt",
            )
            with torch.inference_mode():
                out = model(
                    image_embeddings=emb,
                    input_boxes=prompt["input_boxes"],
                    multimask_output=self.multimask_output,
                )
            masks = proc.post_process_masks(out.pred_masks, original_sizes)[0]  # (n_boxes, n_masks, H, W)
            ious = out.iou_scores[0]  # (n_boxes, n_masks)
        except ProviderError:
            raise
        except Exception as exc:  # surface as explicit provider failure
            raise ProviderError("SEGMENTATION_FAILED", f"SAM inference failed: {type(exc).__name__}") from exc
        results = []
        for b in range(len(boxes)):
            k = int(torch.argmax(ious[b])) if self.multimask_output else 0
            results.append(SamMask(mask=masks[b, k].numpy().astype(bool), predicted_iou=float(ious[b, k])))
        return results
