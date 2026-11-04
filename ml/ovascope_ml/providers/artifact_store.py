"""Local filesystem artifact store with opaque keys.

Keys look like `<namespace>/<random hex>`; namespaces are validated, so a key can
never resolve outside the store root (no path traversal). Paths are never
returned to API clients.
"""

from __future__ import annotations

import re
import secrets
import shutil
from pathlib import Path

from ovascope_ml.providers.base import ArtifactStore, ProviderStatus

_NS = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_OBJ = re.compile(r"^[a-f0-9]{32}$")


class InvalidArtifactKey(ValueError):
    pass


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            kind="artifact_store",
            provider_id="local-filesystem",
            available=self.root.exists(),
            execution="local",
            details={"public": False, "retention": "deleted with the study"},
        )

    def _path(self, key: str) -> Path:
        parts = key.split("/")
        if len(parts) != 2 or not _NS.match(parts[0]) or not _OBJ.match(parts[1]):
            raise InvalidArtifactKey("malformed artifact key")
        path = (self.root / parts[0] / parts[1]).resolve()
        if self.root not in path.parents:
            raise InvalidArtifactKey("artifact key escapes store root")
        return path

    def put(self, namespace: str, data: bytes, media_type: str) -> str:
        if not _NS.match(namespace):
            raise InvalidArtifactKey("malformed namespace")
        key = f"{namespace}/{secrets.token_hex(16)}"
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        try:
            return self._path(key).exists()
        except InvalidArtifactKey:
            return False

    def delete_namespace(self, namespace: str) -> int:
        if not _NS.match(namespace):
            raise InvalidArtifactKey("malformed namespace")
        d = (self.root / namespace).resolve()
        if self.root not in d.parents or not d.exists():
            return 0
        n = sum(1 for p in d.iterdir() if p.is_file())
        shutil.rmtree(d)
        return n
