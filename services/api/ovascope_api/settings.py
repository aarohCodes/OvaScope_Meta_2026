"""API settings (environment variables prefixed OVASCOPE_). Secrets stay server-side."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ovascope_ml import paths


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OVASCOPE_", env_file=".env", extra="ignore")

    deployment: Literal["local", "public"] = "local"
    api_token: SecretStr | None = None
    database_url: str = Field(default_factory=lambda: f"sqlite:///{(paths.data_dir() / 'ovascope.db').as_posix()}")
    artifact_dir: str = Field(default_factory=lambda: str(paths.data_dir() / "artifact_store"))
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
