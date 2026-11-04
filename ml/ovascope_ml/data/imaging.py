"""Defensive image decoding shared by the dataset tools and the upload API.

Checks performed, in order:
  1. Byte-size limit (before any decoding).
  2. Magic-byte sniffing: only PNG and JPEG signatures are accepted.
  3. Pillow header parse with a pixel-count ceiling (decompression-bomb guard).
  4. Declared format must match the sniffed signature; multi-frame images are rejected.
  5. Full decode (`load()`), so truncated or corrupt files fail here, not later.
  6. Dimension limits.

Decoded images are converted to 8-bit RGB. EXIF orientation is NOT applied:
ultrasound frames are stored as acquired, and silently rotating them would
desynchronise masks and overlays.
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"


class InputErrorCode(str, Enum):
    EMPTY_FILE = "EMPTY_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    FORMAT_MISMATCH = "FORMAT_MISMATCH"
    DECOMPRESSION_BOMB = "DECOMPRESSION_BOMB"
    CORRUPT_IMAGE = "CORRUPT_IMAGE"
    MULTI_FRAME = "MULTI_FRAME"
    DIMENSIONS_TOO_SMALL = "DIMENSIONS_TOO_SMALL"
    DIMENSIONS_TOO_LARGE = "DIMENSIONS_TOO_LARGE"


class ImageInputError(ValueError):
    def __init__(self, code: InputErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ImageLimits:
    max_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 40_000_000
    min_side: int = 64
    max_side: int = 8192


@dataclass(frozen=True)
class DecodedImage:
    rgb: np.ndarray  # (H, W, 3) uint8
    format: str  # "PNG" | "JPEG"
    width: int
    height: int
    mode_original: str


def sniff_format(data: bytes) -> str | None:
    if data.startswith(PNG_SIGNATURE):
        return "PNG"
    if data.startswith(JPEG_SIGNATURE):
        return "JPEG"
    return None


def decode_image(data: bytes, limits: ImageLimits | None = None) -> DecodedImage:
    limits = limits or ImageLimits()
    if not data:
        raise ImageInputError(InputErrorCode.EMPTY_FILE, "The file is empty.")
    if len(data) > limits.max_bytes:
        raise ImageInputError(
            InputErrorCode.FILE_TOO_LARGE,
            f"File exceeds the {limits.max_bytes // (1024 * 1024)} MB upload limit.",
        )
    sniffed = sniff_format(data)
    if sniffed is None:
        raise ImageInputError(
            InputErrorCode.UNSUPPORTED_FORMAT,
            "Only PNG and JPEG still images are supported in this version.",
        )

    # Pillow raises DecompressionBombError above 2x MAX_IMAGE_PIXELS and warns above 1x.
    # We treat the warning as an error too, and apply our own explicit ceiling.
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = limits.max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            try:
                with Image.open(io.BytesIO(data)) as im:
                    declared = im.format
                    width, height = im.size
                    if width * height > limits.max_pixels:
                        raise ImageInputError(
                            InputErrorCode.DECOMPRESSION_BOMB,
                            "Decoded image would exceed the pixel limit.",
                        )
                    if declared != sniffed:
                        raise ImageInputError(
                            InputErrorCode.FORMAT_MISMATCH,
                            "File signature does not match the declared image format.",
                        )
                    if getattr(im, "n_frames", 1) > 1:
                        raise ImageInputError(
                            InputErrorCode.MULTI_FRAME,
                            "Multi-frame images (animation, video, cine) are not supported.",
                        )
                    mode = im.mode
                    im.load()
                    rgb = np.asarray(_to_rgb8(im), dtype=np.uint8)
            except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                raise ImageInputError(
                    InputErrorCode.DECOMPRESSION_BOMB, "Decoded image would exceed the pixel limit."
                ) from exc
            except ImageInputError:
                raise
            except Exception as exc:  # Pillow raises many types for corrupt data
                raise ImageInputError(
                    InputErrorCode.CORRUPT_IMAGE, "The image could not be decoded."
                ) from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    if min(width, height) < limits.min_side:
        raise ImageInputError(
            InputErrorCode.DIMENSIONS_TOO_SMALL,
            f"Image is smaller than {limits.min_side} px on one side.",
        )
    if max(width, height) > limits.max_side:
        raise ImageInputError(
            InputErrorCode.DIMENSIONS_TOO_LARGE,
            f"Image is larger than {limits.max_side} px on one side.",
        )
    return DecodedImage(rgb=rgb, format=sniffed, width=width, height=height, mode_original=mode)


def _to_rgb8(im: Image.Image) -> Image.Image:
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        scaled = np.zeros_like(arr) if hi <= lo else (arr - lo) / (hi - lo) * 255.0
        return Image.fromarray(scaled.astype(np.uint8)).convert("RGB")
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        background = Image.new("RGB", im.size, (0, 0, 0))
        rgba = im.convert("RGBA")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return im.convert("RGB")


def load_rgb_file(path: str | os.PathLike[str], limits: ImageLimits | None = None) -> DecodedImage:
    with open(path, "rb") as fh:
        return decode_image(fh.read(), limits)


def load_mask_file(path: str | os.PathLike[str]) -> np.ndarray:
    """Load a binary lesion mask: any non-zero pixel (in any channel) is foreground."""
    with Image.open(path) as im:
        im.load()
        arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., :3].max(axis=-1)
    return arr > 0
