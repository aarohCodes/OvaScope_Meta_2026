"""Content hashing for duplicate and near-duplicate detection.

Exact duplicates: SHA-256 of file bytes and of decoded pixels.
Near duplicates: 64-bit DCT perceptual hash (pHash) and difference hash (dHash);
pairs within a Hamming-distance threshold on BOTH hashes are flagged as
*suspected* near duplicates. This is a heuristic: it neither proves that two
images come from one patient nor proves that they do not.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
from PIL import Image
from scipy.fft import dctn


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_pixels(rgb: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(rgb.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(rgb).tobytes())
    return h.hexdigest()


def _gray(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    im = Image.fromarray(rgb).convert("L").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float64)


def phash64(rgb: np.ndarray) -> int:
    g = _gray(rgb, (32, 32))
    coeffs = dctn(g, norm="ortho")[:8, :8]
    flat = coeffs.flatten()
    median = np.median(flat[1:])  # exclude DC term
    bits = flat > median
    return _bits_to_int(bits)


def dhash64(rgb: np.ndarray) -> int:
    g = _gray(rgb, (9, 8))
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return _bits_to_int(bits)


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for b in bits.astype(bool).tolist():
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def near_duplicate_pairs(
    phashes: list[int], dhashes: list[int], max_phash: int = 6, max_dhash: int = 8
) -> list[tuple[int, int, int, int]]:
    """Return (i, j, phash_distance, dhash_distance) for suspected near duplicates."""
    n = len(phashes)
    p = np.array(phashes, dtype=np.uint64)
    d = np.array(dhashes, dtype=np.uint64)
    pairs: list[tuple[int, int, int, int]] = []
    for i in range(n - 1):
        px = np.bitwise_xor(p[i + 1 :], p[i])
        dx = np.bitwise_xor(d[i + 1 :], d[i])
        pd = _popcount64(px)
        dd = _popcount64(dx)
        hits = np.nonzero((pd <= max_phash) & (dd <= max_dhash))[0]
        for k in hits.tolist():
            pairs.append((i, i + 1 + k, int(pd[k]), int(dd[k])))
    return pairs


def _popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    as_bytes = x.view(np.uint8).reshape(-1, 8)
    return np.unpackbits(as_bytes, axis=1).sum(axis=1)


def connected_components(n: int, pairs: list[tuple[int, int, int, int]]) -> list[int]:
    """Union-find over suspected near-duplicate pairs; returns a cluster id per item."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, *_ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = [find(i) for i in range(n)]
    remap: dict[int, int] = {}
    return [remap.setdefault(r, len(remap)) for r in roots]
