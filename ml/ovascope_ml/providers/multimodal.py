"""Optional Meta multimodal / language provider: interface only, disabled.

Checked 2026-11-04: no Meta Llama API credentials, no documented event endpoint,
and no local Llama weights are configured in this environment. Endpoint URLs,
model identifiers and auth schemes are deliberately NOT guessed.

Report text is produced by a deterministic template (`inference.report`). If a
documented provider is configured later, it may only rephrase verified
structured results, its output must validate against an allow-list schema, and
the template remains the fallback.
"""

from __future__ import annotations

import os

from ovascope_ml.providers.base import OptionalMultimodalProvider, ProviderStatus, ProviderUnavailable

REQUIRED_CONFIG = (
    "OVASCOPE_MULTIMODAL_PROVIDER=meta_model_api",
    "MODEL_API_KEY (from the Meta Model API dashboard, dev.meta.ai)",
    "OVASCOPE_EXTERNAL_TRANSMISSION=structured_results_only",
    "OVASCOPE_MONTHLY_PROVIDER_BUDGET_USD > 0 (see .env.example)",
)


class DisabledMultimodalProvider(OptionalMultimodalProvider):
    def status(self) -> ProviderStatus:
        requested = os.environ.get("OVASCOPE_MULTIMODAL_PROVIDER")
        reason = (
            f"Provider '{requested}' requested, but no documented adapter for it exists in this build."
            if requested
            else "No multimodal/language provider is configured."
        )
        return ProviderStatus(
            kind="multimodal",
            provider_id="none",
            available=False,
            execution="none",
            reason_unavailable=reason,
            action_required="Optional. To enable Meta Muse Spark narratives set: " + "; ".join(REQUIRED_CONFIG),
            details={"report_text": "deterministic template"},
        )

    def summarize_structured_result(self, payload: dict[str, object]) -> str:
        raise ProviderUnavailable("multimodal provider disabled")
