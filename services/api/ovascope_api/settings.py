"""API settings (environment variables prefixed OVASCOPE_). Secrets stay server-side."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from ovascope_ml import paths


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OVASCOPE_", env_file=".env", extra="ignore")

    deployment: Literal["local", "public"] = "local"
    api_token: SecretStr | None = None
    database_url: str = Field(default_factory=lambda: f"sqlite:///{(paths.data_dir() / 'ovascope.db').as_posix()}")
    artifact_dir: str = Field(default_factory=lambda: str(paths.data_dir() / "artifact_store"))
    # NoDecode: let the validator split "a,b" instead of the env source JSON-decoding it.
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    max_upload_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    retention_hours: int = 24
    inline_worker: bool = True
    auto_migrate: bool = True
    uploads_per_minute: int = 30
    max_queued_jobs: int = 20
    job_max_attempts: int = 2
    serve_eval_images: bool = False
    external_transmission_allowed: bool = False  # no authorised hosted provider exists; fail closed
    monthly_provider_budget_usd: float = 0.0  # hosted providers disabled; budget cap enforced as zero

    @field_validator("api_token", mode="before")
    @classmethod
    def _empty_token_is_none(cls, v: object) -> object:
        # Compose passes unset variables as "" (OVASCOPE_API_TOKEN=${OVASCOPE_API_TOKEN:-}); treat as no token.
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split(cls, v: object) -> object:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    def assert_safe(self) -> None:
        if self.deployment == "public":
            if self.api_token is None or len(self.api_token.get_secret_value()) < 24:
                raise RuntimeError(
                    "Refusing to start: OVASCOPE_DEPLOYMENT=public requires OVASCOPE_API_TOKEN (>= 24 chars). "
                    "This prototype is local-only without access control."
                )
            if any(o.strip() == "*" for o in self.cors_origins):
                raise RuntimeError("Refusing to start: wildcard CORS is not allowed in public deployment.")


ENV_KEYS_PREFIX = "OVASCOPE_"
ENV_KEYS_EXTRA = {"MODEL_API_KEY"}  # Meta Model API key name used by Meta's docs


def load_env_file(path: Path | None = None) -> list[str]:
    """Load allow-listed keys from the repo's .env into os.environ (existing values win).

    Returns the key NAMES loaded (never values). Only OVASCOPE_* keys and MODEL_API_KEY are read.
    """
    env_path = path or Path(os.environ.get("OVASCOPE_ENV_FILE", str(paths.REPO_ROOT / ".env")))
    if not env_path.is_file():
        return []
    loaded = []
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not (key.startswith(ENV_KEYS_PREFIX) or key in ENV_KEYS_EXTRA) or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value
        loaded.append(key)
    return loaded


@lru_cache
def get_settings() -> Settings:
    load_env_file()
    return Settings()
