"""Frozen image feature extractors.

* Dinov2Extractor  - Meta DINOv2 (timm packaging), local weights only.
* ResNet50Extractor - supervised ImageNet ResNet-50 (torchvision); comparator without DINOv2.
* HandcraftedExtractor - intensity/gradient statistics; no learned representation.

All extractors consume the same letterboxed input so comparisons are fair.
Weights are never downloaded here; missing weights raise WeightsUnavailable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ovascope_ml.models.geometry import Letterbox
from ovascope_ml.models.registry import WeightSpec, get


class WeightsUnavailable(RuntimeError):
    def __init__(self, message: str, action: str) -> None:
        super().__init__(message)
        self.action = action


def cached_weight_path(spec: WeightSpec, filename: str) -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    found = try_to_load_from_cache(spec.repo_id, filename, revision=spec.revision)
    return Path(found) if isinstance(found, str) else None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DinoOutput:
    cls: np.ndarray  # (D,)
    patch_mean: np.ndarray  # (D,)
    patches: np.ndarray  # (g, g, D) float16
    letterbox: Letterbox

    def global_features(self) -> np.ndarray:
        return np.concatenate([self.cls, self.patch_mean]).astype(np.float32)


class Dinov2Extractor:
    def __init__(self, key: str = "dinov2_vits14", img_size: int = 448, threads: int | None = None) -> None:
        self.spec = get(key)
        self.img_size = img_size
        self._model: Any = None
        self._threads = threads
        self.weights_path = cached_weight_path(self.spec, "model.safetensors")

    @property
    def model_id(self) -> str:
        return f"{self.spec.repo_id}@{self.spec.revision[:12]}"

    def available(self) -> tuple[bool, str | None]:
        try:
            import timm  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False, "PyTorch/timm not installed in this environment (pip install -e '.[ml]')"
        if self.weights_path is None:
            return False, f"DINOv2 weights not in local cache; run `ovascope-ml fetch-models` ({self.spec.repo_id})"
        return True, None

    def _load(self):
        if self._model is not None:
            return self._model
        ok, why = self.available()
        if not ok:
            raise WeightsUnavailable(why or "unavailable", "ovascope-ml fetch-models")
        import timm
        import torch

        if self._threads:
            torch.set_num_threads(self._threads)
        model: Any = timm.create_model(
            self.spec.extra["timm_name"],
            pretrained=True,
            img_size=self.img_size,
            num_classes=0,
            pretrained_cfg_overlay={"source": "", "file": str(self.weights_path)},
        )
        model.eval()
        self._mean = np.asarray(model.pretrained_cfg["mean"], dtype=np.float32)
        self._std = np.asarray(model.pretrained_cfg["std"], dtype=np.float32)
        self._model = model
        return model

    @property
    def grid(self) -> int:
        return self.img_size // 14

    def letterbox_for(self, rgb: np.ndarray) -> Letterbox:
        return Letterbox(src_w=rgb.shape[1], src_h=rgb.shape[0], size=self.img_size)

    def _tensor(self, canvases: list[np.ndarray]):
        import torch

        arr = np.stack(canvases).astype(np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())

    def extract(self, images: list[np.ndarray]) -> list[DinoOutput]:
        import torch

        model = self._load()
        boxes = [self.letterbox_for(im) for im in images]
        x = self._tensor([lb.apply(im) for lb, im in zip(boxes, images, strict=True)])
        with torch.inference_mode():
            tokens = model.forward_features(x).float().numpy()
        n_prefix = model.num_prefix_tokens
        g = self.grid
        outs = []
        for i, lb in enumerate(boxes):
            cls = tokens[i, 0]
            patches = tokens[i, n_prefix:].reshape(g, g, -1)
            outs.append(
                DinoOutput(
                    cls=cls,
                    patch_mean=patches.reshape(-1, patches.shape[-1]).mean(0),
                    patches=patches.astype(np.float16),
                    letterbox=lb,
                )
            )
        return outs


class ResNet50Extractor:
    """Supervised ImageNet ResNet-50 (torchvision IMAGENET1K_V2), global-average-pooled."""

    model_id = "torchvision/resnet50.IMAGENET1K_V2"

    def __init__(self, img_size: int = 448) -> None:
        self.img_size = img_size
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)  # downloads to the torch hub cache with hash check
        model.fc = torch.nn.Identity()
        model.eval()
        self._model = model
        return model

    def extract(self, images: list[np.ndarray]) -> np.ndarray:
        import torch

        model = self._load()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        canv = [Letterbox(im.shape[1], im.shape[0], self.img_size).apply(im) for im in images]
        arr = (np.stack(canv).astype(np.float32) / 255.0 - mean) / std
        x = torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())
        with torch.inference_mode():
            return model(x).float().numpy()


class HandcraftedExtractor:
    """Grey-level intensity and gradient statistics plus a coarse thumbnail (no learning)."""

    model_id = "handcrafted-intensity-gradient-v1"

    def __init__(self, img_size: int = 224) -> None:
        self.img_size = img_size

    def extract(self, images: list[np.ndarray]) -> np.ndarray:
        feats = []
        for im in images:
            lb = Letterbox(im.shape[1], im.shape[0], self.img_size)
            g = lb.apply(im).astype(np.float32).mean(axis=-1) / 255.0
            valid = lb.valid_region()
            vals = g[valid]
            hist = np.histogram(vals, bins=32, range=(0, 1))[0] / max(len(vals), 1)
            gy, gx = np.gradient(g)
            mag = np.hypot(gx, gy)[valid]
            ghist = np.histogram(mag, bins=16, range=(0, 0.5))[0] / max(len(mag), 1)
            thumb = g.reshape(16, self.img_size // 16, 16, self.img_size // 16).mean(axis=(1, 3)).ravel()
            stats = [vals.mean(), vals.std(), np.percentile(vals, 10), np.percentile(vals, 90), mag.mean()]
            feats.append(np.concatenate([hist, ghist, thumb, stats]).astype(np.float32))
        return np.stack(feats)
