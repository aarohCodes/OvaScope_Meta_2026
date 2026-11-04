"""Optional Modal GPU adapter for feature extraction on an AUTHORISED dataset.

Status: written against Modal's documented Python API (modal.App, modal.Image,
modal.Volume, @app.function(gpu=...), @app.local_entrypoint). NOT executed in the
development environment: no Modal credentials were available (2026-11-04).

Governance gate: running this uploads images to a third-party cloud. The MMOTU
images used in this prototype have no published data licence, so external
transmission is NOT authorised and this adapter refuses to run for them. It runs
only when BOTH are true:
  * OVASCOPE_EXTERNAL_TRANSMISSION_AUTHORISED=yes  (a human has written authorisation)
  * the manifest's source.external_api_transmission_authorized is true

Usage (after `modal setup`):
  OVASCOPE_EXTERNAL_TRANSMISSION_AUTHORISED=yes modal run infra/modal/ovascope_modal.py --manifest path/to/manifest.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2]
app = modal.App("ovascope-features")
weights = modal.Volume.from_name("ovascope-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REPO / "requirements" / "lock.txt"), extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_python_source("ovascope_ml")
)


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/root/.cache/huggingface": weights}, max_containers=1)
def extract_dino(batch: list[bytes], key: str = "dinov2_vits14", img_size: int = 448) -> list[list[float]]:
    from ovascope_ml.data.imaging import decode_image
    from ovascope_ml.models.backbones import Dinov2Extractor

    ext = Dinov2Extractor(key, img_size=img_size)
    outs = ext.extract([decode_image(b).rgb for b in batch])
    return [o.global_features().tolist() for o in outs]


@app.local_entrypoint()
def main(manifest: str, max_images: int = 200) -> None:
    if os.environ.get("OVASCOPE_EXTERNAL_TRANSMISSION_AUTHORISED") != "yes":
        sys.exit("Refusing: external transmission is not authorised (set OVASCOPE_EXTERNAL_TRANSMISSION_AUTHORISED=yes only with written authorisation).")
    m = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not m["source"].get("external_api_transmission_authorized"):
        sys.exit("Refusing: the dataset manifest does not authorise external API transmission.")
    root = Path(os.environ["OVASCOPE_DATASET_ROOT"])
    recs = m["records"][:max_images]  # spending cap: bounded number of images per run
    feats = extract_dino.remote([(root / r["image_path"]).read_bytes() for r in recs])
    out = Path("modal_features.json")
    out.write_text(json.dumps({"image_ids": [r["image_id"] for r in recs], "features": feats}), encoding="utf-8")
    print(f"wrote {out} ({len(feats)} images)")
