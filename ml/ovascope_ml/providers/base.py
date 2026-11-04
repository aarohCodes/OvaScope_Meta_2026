"""Typed provider interfaces.

A provider either works, with an actual model identifier, or reports itself
unavailable with the exact missing requirement. There is no simulated mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ProviderKind = Literal["segmentation", "classifier", "multimodal", "artifact_store", "representation"]


@dataclass(frozen=True)
class ProviderStatus:
    kind: ProviderKind
    provider_id: str
    available: bool
    model_id: str | None = None
    model_version: str | None = None
    license: str | None = None
    meta_model: bool = False
    execution: Literal["local", "hosted", "none"] = "local"
    reason_unavailable: str | None = None
    action_required: str | None = None
    details: dict[str, object] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """A provider was available but failed while processing."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ProviderTimeout(ProviderError):
    def __init__(self, message: str = "provider exceeded its time limit") -> None:
        super().__init__("PROVIDER_TIMEOUT", message)


class ProviderUnavailable(ProviderError):
    def __init__(self, message: str) -> None:
        super().__init__("PROVIDER_UNAVAILABLE", message)


@dataclass
class SegmentationOutcome:
    status: Literal["succeeded", "no_candidate_region", "provider_unavailable", "provider_failed", "timeout", "not_run"]
    provider_id: str
    model_id: str | None
    prompt_source: str | None
    mask: np.ndarray | None = None  # (H, W) bool, produced by the SAM-family model only
    prompt_box: tuple[int, int, int, int] | None = None
    localizer_grid: np.ndarray | None = None
    localizer_max_score: float | None = None
    sam_predicted_iou: float | None = None
    message: str | None = None


class SegmentationProvider(ABC):
    @abstractmethod
    def status(self) -> ProviderStatus: ...

    @abstractmethod
    def segment(self, rgb: np.ndarray, patches: np.ndarray | None, letterbox: object | None) -> SegmentationOutcome:
        """Automatic segmentation. Must never fabricate a mask on failure."""


class ImageClassifier(ABC):
    @abstractmethod
    def status(self) -> ProviderStatus: ...


class OptionalMultimodalProvider(ABC):
    """Narrow, evidence-bounded text tasks only (e.g. summarising verified structured results).

    Must never produce or alter scores, infer stage, or recommend care.
    """

    @abstractmethod
    def status(self) -> ProviderStatus: ...

    @abstractmethod
    def summarize_structured_result(self, payload: dict[str, object]) -> str: ...


class ArtifactStore(ABC):
    @abstractmethod
    def status(self) -> ProviderStatus: ...

    @abstractmethod
    def put(self, namespace: str, data: bytes, media_type: str) -> str:
        """Store bytes; return an opaque key."""

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def delete_namespace(self, namespace: str) -> int:
        """Delete every object in a namespace; return the number removed."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...
