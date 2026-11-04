"""Fetch the OTU_2d subset of MMOTU from a public Hugging Face mirror.

Provenance policy (see docs/DATASET_READINESS.md):
  * The authoritative release is the authors' Google Drive folder linked from
    https://github.com/cv516Buaa/MMOTU_DS2Net . Bulk download from Drive was
    rate-limited, so this script uses the third-party mirror `norayao/MMOTU`
    pinned to an exact revision purely as a transport.
  * Every LFS file is checked against the SHA-256 recorded by the mirror, and
    `python -m ovascope_ml.cli verify-mirror` compares files fetched directly
    from the official Drive folder against this copy. The mirror's own license
    tag is NOT treated as a data license.
  * Only OTU_2d images, annotations and the official split/label text files are
    fetched. The mirror uploader's own summary files are excluded.
  * No credentials are sent. Requests are sequential and honour HTTP 429.

Usage:  python scripts/fetch_mmotu_mirror.py <output_dir>
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path, PurePosixPath

import requests

REPO_ID = "norayao/MMOTU"
REVISION = "d29379c60ac4072cfbf9c6f960505d43f98b3f0d"
API = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/{REVISION}"
RESOLVE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}"
WANTED_DIRS = ("OTU_2d/images", "OTU_2d/annotations")
WANTED_FILES = ("OTU_2d/train.txt", "OTU_2d/train_cls.txt", "OTU_2d/val.txt", "OTU_2d/val_cls.txt")
# `{id}_binary_binary.PNG` files are not read by the official code (which uses
# `{id}_binary.PNG` as the lesion mask); skip them to reduce request volume.
SKIP_SUFFIX = "_binary_binary.PNG"
PAUSE_SECONDS = 0.35
USER_AGENT = "ovascope-dataset-fetch/0.1"


def _get(session: requests.Session, url: str, **kw: object) -> requests.Response:
    for _ in range(60):
        resp = session.get(url, timeout=60, **kw)  # type: ignore[arg-type]
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        wait = int(resp.headers.get("Retry-After", "60") or 60)
        print(f"429 for {url.rsplit('/', 1)[-1]}; sleeping {wait}s", flush=True)
        time.sleep(min(max(wait, 5), 300))
    raise RuntimeError(f"giving up on {url}")


def _list_tree(session: requests.Session, path: str) -> list[dict]:
    entries: list[dict] = []
    url: str | None = f"{API}/{path}?recursive=false&expand=false"
    while url:
        resp = _get(session, url)
        entries.extend(resp.json())
        nxt = resp.links.get("next", {}).get("url")
        url = nxt
    return entries


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    remote: dict[str, str | None] = {f: None for f in WANTED_FILES}
    for d in WANTED_DIRS:
        for e in _list_tree(session, d):
            if e.get("type") != "file" or e["path"].endswith(SKIP_SUFFIX):
                continue
            remote[e["path"]] = (e.get("lfs") or {}).get("oid")
    print(f"{len(remote)} files listed at revision {REVISION}", flush=True)

    fetched = verified = 0
    mismatches: list[str] = []
    for rel, oid in sorted(remote.items()):
        rp = PurePosixPath(rel)
        if ".." in rp.parts or rp.is_absolute():
            raise ValueError(f"refusing suspicious path {rel!r}")
        dest = out.joinpath(*rp.parts)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            resp = _get(session, f"{RESOLVE}/{rel}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            fetched += 1
            time.sleep(PAUSE_SECONDS)
            if fetched % 200 == 0:
                print(f"fetched {fetched}", flush=True)
        if oid is not None:
            if _sha256(dest) == oid:
                verified += 1
            else:
                mismatches.append(rel)

    record = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "files_listed": len(remote),
        "files_fetched_this_run": fetched,
        "lfs_sha256_verified": verified,
        "lfs_sha256_mismatches": mismatches,
        "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "MIRROR_PROVENANCE.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
