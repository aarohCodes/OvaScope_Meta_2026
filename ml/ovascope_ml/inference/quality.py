"""Input checks, kept separate by kind.

Technical checks decide whether the file can be processed at all. Image-quality and
content checks are *warnings*: none of them is evidence of clinical adequacy, and
none is validated against expert quality ratings.
"""

from __future__ import annotations

import numpy as np

from ovascope_ml.inference.result_schema import QualityCheck

# Observed range of the MMOTU OTU_2d development images (paper: 302-1135 x 226-794).
DEV_WIDTH_RANGE = (302, 1135)
DEV_HEIGHT_RANGE = (226, 794)


def technical_pass(fmt: str, width: int, height: int) -> list[QualityCheck]:
    return [
        QualityCheck(
            id="decode",
            category="technical",
            status="pass",
            message=f"{fmt} decoded with size and pixel-count limits enforced.",
        ),
        QualityCheck(
            id="dimensions",
            category="technical",
            status="pass",
            message=f"{width} x {height} px is within the accepted range.",
        ),
    ]


def image_checks(rgb: np.ndarray) -> list[QualityCheck]:
    gray = rgb.astype(np.float32).mean(axis=-1)
    h, w = gray.shape
    checks: list[QualityCheck] = []
    std = float(gray.std())
    checks.append(
        QualityCheck(
            id="dynamic_range",
            category="image_quality",
            status="warn" if std < 12 else "pass",
            value=std,
            message=(
                "Very low intensity variation (possible blank or saturated frame)."
                if std < 12
                else "Intensity variation is not unusually low."
            )
            + " Heuristic; not a clinical adequacy measure.",
        )
    )
    dark = float((gray < 8).mean())
    checks.append(
        QualityCheck(
            id="near_black_fraction",
            category="image_quality",
            status="warn" if dark > 0.85 else "pass",
            value=dark,
            message=("Most of the frame is near-black." if dark > 0.85 else "Frame is not predominantly black.")
            + " Heuristic.",
        )
    )
    in_range = DEV_WIDTH_RANGE[0] <= w <= DEV_WIDTH_RANGE[1] and DEV_HEIGHT_RANGE[0] <= h <= DEV_HEIGHT_RANGE[1]
    checks.append(
        QualityCheck(
            id="frame_size_vs_development",
            category="distribution_shift",
            status="pass" if in_range else "warn",
            message=(
                "Frame size is within the range seen in development images."
                if in_range
                else f"Frame size {w}x{h} is outside the development range "
                f"{DEV_WIDTH_RANGE[0]}-{DEV_WIDTH_RANGE[1]} x {DEV_HEIGHT_RANGE[0]}-{DEV_HEIGHT_RANGE[1]} px."
            ),
        )
    )
    arr = rgb.astype(np.int16)
    chroma = float(((arr.max(-1) - arr.min(-1)) > 40).mean())
    checks.append(
        QualityCheck(
            id="colour_content",
            category="content",
            status="info" if chroma > 0.01 else "pass",
            value=chroma,
            message=(
                "Colour pixels present (colour Doppler and/or coloured annotations). Coloured annotations "
                "are a known shortcut risk in the training data."
                if chroma > 0.01
                else "Predominantly greyscale frame."
            ),
        )
    )
    return checks
