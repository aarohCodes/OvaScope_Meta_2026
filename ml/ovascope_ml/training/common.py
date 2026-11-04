"""Shared training utilities: config, feature store, label-safe record views."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ovascope_ml import paths
from ovascope_ml.data.label_policy import MMOTU_CLASSES
from ovascope_ml.data.manifest import Manifest

CLASS_NAMES = [c.normalized for c in MMOTU_CLASSES]
HGS_INDEX = CLASS_NAMES.index("high_grade_serous")


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    cfg["_config_text"] = path.read_text(encoding="utf-8")
    return cfg


def experiment_dir(cfg: dict[str, Any]) -> Path:
    d = paths.experiments_dir() / cfg["experiment_id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_dir(cfg: dict[str, Any], stage: str) -> Path:
    d = paths.processed_dir() / "pipeline" / cfg["experiment_id"] / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_manifest(cfg: dict[str, Any]) -> Manifest:
    return Manifest.load(paths.REPO_ROOT / cfg["dataset"]["manifest"])


@dataclass(frozen=True)
class DevView:
    """Development records WITH labels (training and model selection)."""

    image_ids: list[str]
    image_paths: list[str]
    labels: np.ndarray
    mask_paths: list[str | None]


@dataclass(frozen=True)
class TestInputs:
    """Test records WITHOUT labels or masks: only what an inference system would see."""

    image_ids: list[str]
    image_paths: list[str]


def dev_view(m: Manifest) -> DevView:
    recs = m.by_split("development")
    return DevView(
        image_ids=[r.image_id for r in recs],
        image_paths=[r.image_path for r in recs],
        labels=np.array([CLASS_NAMES.index(r.label_normalized) for r in recs]),
        mask_paths=[r.mask_path_if_available for r in recs],
    )


def test_inputs(m: Manifest) -> TestInputs:
    recs = m.by_split("test")
    return TestInputs(image_ids=[r.image_id for r in recs], image_paths=[r.image_path for r in recs])


class FeatureStore:
    """Arrays aligned to an image-id order, with a provenance sidecar."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def has(self, name: str) -> bool:
        return (self.dir / f"{name}.npy").exists() and (self.dir / f"{name}.json").exists()

    def save(self, name: str, array: np.ndarray, image_ids: list[str], provenance: dict[str, Any]) -> None:
        np.save(self.dir / f"{name}.npy", array, allow_pickle=False)
        meta = {
            "image_ids": image_ids,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **provenance,
        }
        (self.dir / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self, name: str, mmap: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        arr = np.load(self.dir / f"{name}.npy", mmap_mode="r" if mmap else None, allow_pickle=False)
        meta = json.loads((self.dir / f"{name}.json").read_text(encoding="utf-8"))
        return arr, meta

    def rows(self, name: str, image_ids: list[str], mmap: bool = False) -> np.ndarray:
        arr, meta = self.load(name, mmap=mmap)
        index = {iid: i for i, iid in enumerate(meta["image_ids"])}
        return np.asarray(arr[[index[i] for i in image_ids]])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
