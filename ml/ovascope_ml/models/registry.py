"""Pinned registry of third-party model weights used by OvaScope.

Every entry records the exact Hub repository, revision, files and licence that
were checked on 2026-11-04. Inference never downloads weights implicitly:
weights are fetched only by `ovascope-ml fetch-models`, and providers load them
with `local_files_only=True`, reporting "unavailable" when they are missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WeightSpec:
    key: str
    repo_id: str
    revision: str
    files: tuple[str, ...]
    license: str
    publisher: str
    meta_model: bool
    role: str
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)


REGISTRY: dict[str, WeightSpec] = {
    "dinov2_vits14": WeightSpec(
        key="dinov2_vits14",
        repo_id="timm/vit_small_patch14_dinov2.lvd142m",
        revision="4610ca143709d58a633b6397a74412c2c3842454",
        files=("model.safetensors", "config.json"),
        license="apache-2.0",
        publisher="Meta AI (DINOv2), packaged by timm",
        meta_model=True,
        role="frozen image representation + patch tokens for lesion localisation",
        extra={"timm_name": "vit_small_patch14_dinov2.lvd142m"},
    ),
    "dinov2_vitb14": WeightSpec(
        key="dinov2_vitb14",
        repo_id="timm/vit_base_patch14_dinov2.lvd142m",
        revision="4685c99dabffe5affac90bd99dbffd25801ae58d",
        files=("model.safetensors", "config.json"),
        license="apache-2.0",
        publisher="Meta AI (DINOv2), packaged by timm",
        meta_model=True,
        role="optional larger frozen representation",
        extra={"timm_name": "vit_base_patch14_dinov2.lvd142m"},
    ),
    "sam2.1_hiera_tiny": WeightSpec(
        key="sam2.1_hiera_tiny",
        repo_id="facebook/sam2.1-hiera-tiny",
        revision="de431c4043854a71d8101e17995dfe596bf101a5",
        files=("config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json"),
        license="apache-2.0",
        publisher="Meta AI (Segment Anything 2.1)",
        meta_model=True,
        role="box-prompted lesion mask refinement",
    ),
}

# Recorded so the UI and docs can state why it is not used.
NOT_USED = {
    "facebook/sam-vit-base": (
        "Public (Apache-2.0) but no adapter is implemented; SAM 2.1 tiny is the only SAM-family "
        "provider exercised and tested in this prototype."
    ),
    "facebook/sam3": (
        "Gated on the Hugging Face Hub (manual approval), licence 'other', 3.4 GB, and not "
        "supported by the pinned transformers 4.56.2. Not used."
    ),
}


def get(key: str) -> WeightSpec:
    try:
        return REGISTRY[key]
    except KeyError as exc:  # pragma: no cover - configuration error
        raise KeyError(f"unknown weight key {key!r}; known: {sorted(REGISTRY)}") from exc
