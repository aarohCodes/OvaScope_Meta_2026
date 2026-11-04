import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from conftest import geometric_png
from ovascope_ml.data.hashing import (
    connected_components,
    dhash64,
    hamming,
    near_duplicate_pairs,
    phash64,
    sha256_pixels,
)
from ovascope_ml.data.imaging import ImageInputError, ImageLimits, InputErrorCode, decode_image


def _png_header_only(width: int, height: int) -> bytes:
    """A PNG whose IHDR declares a huge size (decompression-bomb shape)."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    idat_data = zlib.compress(b"\x00" * 10)
    idat = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + struct.pack(">I", zlib.crc32(b"IDAT" + idat_data))
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))
    return sig + chunk + idat + iend


def test_valid_png_and_jpeg_decode_to_rgb():
    for fmt in ("PNG", "JPEG"):
        d = decode_image(geometric_png(fmt=fmt))
        assert d.format == fmt and d.rgb.shape == (300, 400, 3) and d.rgb.dtype == np.uint8


@pytest.mark.parametrize(
    "data,code",
    [
        (b"", InputErrorCode.EMPTY_FILE),
        (b"GIF89a" + b"\x00" * 50, InputErrorCode.UNSUPPORTED_FORMAT),
        (b"%PDF-1.4 not an image", InputErrorCode.UNSUPPORTED_FORMAT),
        (b"\x89PNG\r\n\x1a\n" + b"garbage" * 20, InputErrorCode.CORRUPT_IMAGE),
        (b"\xff\xd8\xff" + b"\x00" * 64, InputErrorCode.CORRUPT_IMAGE),
    ],
)
def test_invalid_inputs_rejected_with_typed_codes(data, code):
    with pytest.raises(ImageInputError) as e:
        decode_image(data)
    assert e.value.code == code


def test_oversized_file_rejected_before_decoding():
    with pytest.raises(ImageInputError) as e:
        decode_image(geometric_png(), ImageLimits(max_bytes=100))
    assert e.value.code == InputErrorCode.FILE_TOO_LARGE


def test_decompression_bomb_rejected():
    with pytest.raises(ImageInputError) as e:
        decode_image(_png_header_only(60_000, 60_000))
    assert e.value.code == InputErrorCode.DECOMPRESSION_BOMB


def test_format_mismatch_and_small_images_rejected():
    png = geometric_png()
    jpeg_body = geometric_png(fmt="JPEG")
    with pytest.raises(ImageInputError) as e:
        decode_image(b"\x89PNG\r\n\x1a\n" + jpeg_body[3:])
    assert e.value.code in (InputErrorCode.FORMAT_MISMATCH, InputErrorCode.CORRUPT_IMAGE)
    with pytest.raises(ImageInputError) as e:
        decode_image(geometric_png(32, 32))
    assert e.value.code == InputErrorCode.DIMENSIONS_TOO_SMALL
    assert decode_image(png).width == 400


def test_sixteen_bit_and_alpha_inputs_are_normalised():
    arr16 = (np.linspace(0, 65535, 200 * 100).reshape(100, 200)).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(arr16).save(buf, "PNG")
    d = decode_image(buf.getvalue())
    assert d.rgb.shape == (100, 200, 3) and d.rgb.max() == 255
    rgba = Image.new("RGBA", (120, 90), (255, 0, 0, 0))
    buf = io.BytesIO()
    rgba.save(buf, "PNG")
    assert decode_image(buf.getvalue()).rgb.max() == 0  # transparent -> black background


def test_exact_and_near_duplicate_detection():
    base = decode_image(geometric_png(seed=1)).rgb
    same = base.copy()
    noisy = np.clip(base.astype(int) + np.random.default_rng(2).integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
    other = decode_image(geometric_png(shape="rect", seed=3)).rgb
    assert sha256_pixels(base) == sha256_pixels(same)
    assert sha256_pixels(base) != sha256_pixels(noisy)
    ph = [phash64(x) for x in (base, noisy, other)]
    dh = [dhash64(x) for x in (base, noisy, other)]
    assert hamming(ph[0], ph[1]) <= 6
    pairs = near_duplicate_pairs(ph, dh)
    assert (0, 1) in {(i, j) for i, j, *_ in pairs}
    assert all(2 not in (i, j) for i, j, *_ in pairs)
    clusters = connected_components(3, pairs)
    assert clusters[0] == clusters[1] != clusters[2]
