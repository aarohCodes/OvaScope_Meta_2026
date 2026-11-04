"""Filesystem locations, overridable through environment variables.

Nothing here is ever returned to API clients: the API exposes opaque IDs only.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).resolve() if value else default


def data_dir() -> Path:
    return _env_path("OVASCOPE_DATA_DIR", REPO_ROOT / "data")


def mmotu_root() -> Path:
    """Directory containing `OTU_2d/` (images, annotations, split files)."""
    return _env_path("OVASCOPE_MMOTU_ROOT", data_dir() / "raw" / "mmotu_hf")


def processed_dir() -> Path:
    return _env_path("OVASCOPE_PROCESSED_DIR", data_dir() / "processed")


def experiments_dir() -> Path:
    """Training/evaluation outputs: metrics JSON, per-case CSV, figures data."""
    return _env_path("OVASCOPE_EXPERIMENTS_DIR", REPO_ROOT / "artifacts" / "experiments")


def bundles_dir() -> Path:
    """Exported model bundles used for inference."""
    return _env_path("OVASCOPE_BUNDLES_DIR", REPO_ROOT / "artifacts" / "bundles")


def evaluation_dir() -> Path:
    """Published evaluation artifacts served by the API."""
    return _env_path("OVASCOPE_EVALUATION_DIR", REPO_ROOT / "artifacts" / "evaluation")
