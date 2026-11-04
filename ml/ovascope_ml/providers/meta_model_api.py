"""Meta Model API adapter (Muse Spark) for narrative report text ONLY.

Interface checked against Meta's docs on 2026-11-04 (dev.meta.ai/docs/protocols/chat-completions,
/authentication, /models, /pricing-rate-limits):
  POST {base}/chat/completions   base default https://api.meta.ai/v1
  Authorization: Bearer $MODEL_API_KEY ; body {model, messages, max_completion_tokens, temperature}
  response: choices[0].message.content ; usage.prompt_tokens / completion_tokens
Standard-mode content is "not used to train Meta models"; `-contributor` model IDs "may be used
to train" and are refused here.

Scope and guards (brief section 15):
  * Input is a MINIMISED copy of verified structured results: no image, no study ID, no input hash,
    no timestamps. Scores are rounded to 2 decimals.
  * The model may only rephrase. Its output must be JSON {"summary", "limitations"}, pass a
    diagnostic-language filter, and contain no number absent from the input. Otherwise the
    deterministic template is used and the reason is recorded.
  * Nothing it returns changes a score, a mask, or a status.
  * Requires OVASCOPE_EXTERNAL_TRANSMISSION=structured_results_only, a key, a non-zero monthly
    budget, and per-call consent at the API layer.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ovascope_ml import RESEARCH_NOTICE, paths
from ovascope_ml.providers.base import OptionalMultimodalProvider, ProviderError, ProviderStatus

DEFAULT_BASE = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
DOCS_URL = "https://dev.meta.ai/docs/protocols/chat-completions"
# Standard-mode prices per 1M tokens from dev.meta.ai/docs/pricing-rate-limits (2026-11-04).
PRICE_PER_M = {"muse-spark-1.3": (1.25, 4.25), "muse-spark-1.2": (1.25, 4.25), "muse-spark-1.1": (1.25, 4.25)}
MAX_OUTPUT_TOKENS = 400

FORBIDDEN = (
    "cancer",
    "malignan",
    "tumour is",
    "tumor is",
    "you have",
    "patient has",
    "diagnos",
    "risk",
    "probabilit",
    "likelihood",
    "chance",
    "stage",
    "treat",
    "surgery",
    "surgical",
    "biopsy",
    "recommend",
    "should undergo",
    "follow-up",
    "safe to ignore",
    "%",
)

SYSTEM_PROMPT = (
    "You rewrite verified structured results from a research prototype into a short neutral paragraph "
    "for researchers. Rules: use only facts in the JSON; never add numbers; do not diagnose, do not say "
    "or imply that anyone has or lacks a disease; never mention risk, probability, stage, treatment, "
    "surgery or follow-up; call scores 'model scores'; mention that the result is from a research "
    "prototype evaluated at image level without external validation. Respond with JSON only: "
    '{"summary": "<= 110 words", "limitations": ["<= 3 short items"]}'
)


@dataclass
class Narrative:
    text: str
    limitations: list[str]
    source: str  # "meta-model-api" | "template"
    model_id: str | None
    validated: bool
    fallback_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    sent_payload: dict[str, Any] = field(default_factory=dict)


def minimize(result: dict[str, Any], evaluation_summary: dict[str, Any] | None) -> dict[str, Any]:
    """The only data that may leave the machine: no image, IDs, hashes or timestamps."""
    c = result.get("classification") or {}
    seg = result.get("segmentation") or {}
    out: dict[str, Any] = {
        "notice": RESEARCH_NOTICE,
        "result_status": result.get("status"),
        "classification_target": "eight MMOTU dataset categories (image level)",
        "pathway": c.get("pathway"),
        "highest_scoring_category": c.get("highest_scoring_display")
        if result.get("abstention_status") != "abstained"
        else None,
        "model_scores": [{"category": s["display"], "score": round(float(s["score"]), 2)} for s in c.get("scores", [])],
        "score_semantics": "temperature-scaled model scores, not calibrated probabilities",
        "abstained": result.get("abstention_status") == "abstained",
        "segmentation_status": seg.get("status"),
        "segmentation_source": "SAM 2.1 mask from an automatic DINOv2 box prompt"
        if seg.get("status") == "succeeded"
        else None,
        "warnings": [q["message"] for q in result.get("quality_checks", []) if q.get("status") in ("warn", "fail")][:5],
    }
    if evaluation_summary and evaluation_summary.get("balanced_accuracy"):
        ba = evaluation_summary["balanced_accuracy"]
        out["held_out_balanced_accuracy"] = round(float(ba["point"]), 2)
        out["evaluation_scope"] = (
            "image-level, single dataset, patient-independent evaluation not established, external validation not performed"
        )
    return out


_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _numbers(obj: Any) -> set[str]:
    text = json.dumps(obj)
    found: set[str] = set()
    for n in _NUM.findall(text):
        found.add(n)
        try:
            f = float(n)
            found.update({f"{f:.2f}", f"{f:.1f}", f"{f:g}", str(int(f)) if f.is_integer() else n})
        except ValueError:
            pass
    return found


def validate_output(raw: str, sent: dict[str, Any]) -> tuple[str, list[str]] | str:
    """Return (summary, limitations) if acceptable, else a rejection reason."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    try:
        obj = json.loads(text[text.find("{") : text.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return "output was not valid JSON"
    if not isinstance(obj, dict) or set(obj) != {"summary", "limitations"}:
        return "output keys did not match the schema"
    summary, lims = obj["summary"], obj["limitations"]
    if not isinstance(summary, str) or not (20 <= len(summary) <= 1200):
        return "summary length outside 20-1200 characters"
    if not isinstance(lims, list) or len(lims) > 5 or not all(isinstance(x, str) and len(x) <= 300 for x in lims):
        return "limitations did not match the schema"
    joined = (summary + " " + " ".join(lims)).lower()
    for term in FORBIDDEN:
        if term in joined:
            return f"output contained forbidden term '{term}'"
    allowed = _numbers(sent) | {"8", "2.1", "2", "1"}
    for n in _NUM.findall(summary + " " + " ".join(lims)):
        if n not in allowed:
            return f"output introduced a number not in the input ({n})"
    return summary, lims


def template_narrative(sent: dict[str, Any]) -> tuple[str, list[str]]:
    if sent.get("result_status") == "model_unavailable":
        s = "No trained classifier was configured, so no model scores were produced."
    elif sent.get("abstained"):
        s = "The classifier abstained: the highest model score was below the threshold frozen on development data."
    elif sent.get("highest_scoring_category"):
        s = (
            f"The highest-scoring MMOTU category was {sent['highest_scoring_category']} "
            f"(pathway {sent.get('pathway')}); scores are temperature-scaled model scores, not probabilities."
        )
    else:
        s = f"Result status: {sent.get('result_status')}."
    seg = sent.get("segmentation_status")
    s += f" Segmentation status: {seg}." if seg else ""
    return s + " " + RESEARCH_NOTICE, ["Image-level evaluation on a single dataset; external validation not performed."]


class _Budget:
    def __init__(self, path: Path, monthly_cap_usd: float) -> None:
        self.path, self.cap = path, monthly_cap_usd
        self.lock = threading.Lock()

    def _month(self) -> str:
        return datetime.now(UTC).strftime("%Y-%m")

    def spent(self) -> float:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0.0
        return float(data.get(self._month(), {}).get("usd", 0.0))

    def remaining(self) -> float:
        return max(0.0, self.cap - self.spent())

    def add(self, usd: float) -> None:
        with self.lock:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            m = data.setdefault(self._month(), {"usd": 0.0, "calls": 0})
            m["usd"] = round(m["usd"] + usd, 6)
            m["calls"] += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data), encoding="utf-8")


class MetaModelApiNarrator(OptionalMultimodalProvider):
    provider_id = "meta-model-api"

    def __init__(
        self,
        api_key: str | None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE,
        transmission_authorised: bool = False,
        monthly_budget_usd: float = 0.0,
        usage_path: Path | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.authorised = transmission_authorised
        self.budget = _Budget(usage_path or (paths.data_dir() / "provider_usage.json"), monthly_budget_usd)
        self.timeout_s = timeout_s

    @classmethod
    def from_environment(cls) -> MetaModelApiNarrator:
        return cls(
            api_key=os.environ.get("MODEL_API_KEY") or None,
            model=os.environ.get("OVASCOPE_META_MODEL", DEFAULT_MODEL),
            base_url=os.environ.get("OVASCOPE_META_API_BASE", DEFAULT_BASE),
            transmission_authorised=os.environ.get("OVASCOPE_EXTERNAL_TRANSMISSION") == "structured_results_only",
            monthly_budget_usd=float(os.environ.get("OVASCOPE_MONTHLY_PROVIDER_BUDGET_USD", "0") or 0),
        )

    def _problem(self) -> tuple[str, str] | None:
        if "contributor" in self.model:
            return (
                "Contributor-mode models may use inputs for training and are refused.",
                "Set OVASCOPE_META_MODEL to a Standard-mode model such as muse-spark-1.3",
            )
        if not self.base_url.startswith("https://"):
            return ("Provider base URL must use HTTPS.", "Fix OVASCOPE_META_API_BASE")
        if not self._key:
            return (
                "MODEL_API_KEY is not set.",
                "Create a key in the Meta Model API dashboard and add MODEL_API_KEY to .env",
            )
        if not self.authorised:
            return (
                "External transmission of structured results is not authorised.",
                "Set OVASCOPE_EXTERNAL_TRANSMISSION=structured_results_only after reviewing what is sent",
            )
        if self.budget.remaining() <= 0:
            return (
                "Monthly provider budget is exhausted or zero.",
                "Set OVASCOPE_MONTHLY_PROVIDER_BUDGET_USD (e.g. 5)",
            )
        return None

    def status(self) -> ProviderStatus:
        p = self._problem()
        details = {
            "docs": DOCS_URL,
            "sends": "minimised structured result fields only; never images",
            "monthly_budget_usd": self.budget.cap,
            "spent_this_month_usd": round(self.budget.spent(), 4),
            "consent": "required per call",
        }
        return ProviderStatus(
            kind="multimodal",
            provider_id=self.provider_id,
            available=p is None,
            model_id=self.model,
            meta_model=True,
            execution="hosted",
            license="Meta Model API terms (Standard mode)",
            reason_unavailable=p[0] if p else None,
            action_required=p[1] if p else None,
            details=details,
        )

    def summarize_structured_result(self, payload: dict[str, Any]) -> str:
        return self.narrate(payload).text

    def narrate(self, sent: dict[str, Any]) -> Narrative:
        problem = self._problem()
        if problem:
            raise ProviderError("PROVIDER_UNAVAILABLE", problem[0])
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(sent, ensure_ascii=False)},
            ],
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.2,
        }
        req = urllib.request.Request(  # noqa: S310 - scheme restricted to https in _problem()
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - HTTPS enforced above
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError("PROVIDER_FAILED", f"Meta Model API returned HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError("PROVIDER_TIMEOUT", f"Meta Model API unreachable ({type(exc).__name__})") from None
        usage = data.get("usage") or {}
        pin, pout = PRICE_PER_M.get(self.model, (4.0, 20.0))  # unknown model: assume the most expensive listed
        cost = usage.get("prompt_tokens", 0) / 1e6 * pin + usage.get("completion_tokens", 0) / 1e6 * pout
        self.budget.add(cost)
        try:
            raw = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raw = ""
        checked = validate_output(raw or "", sent)
        if isinstance(checked, str):
            text, lims = template_narrative(sent)
            return Narrative(
                text,
                lims,
                "template",
                self.model,
                validated=False,
                fallback_reason=checked,
                usage=usage,
                estimated_cost_usd=round(cost, 6),
                sent_payload=sent,
            )
        summary, lims = checked
        return Narrative(
            summary,
            lims,
            self.provider_id,
            self.model,
            validated=True,
            usage=usage,
            estimated_cost_usd=round(cost, 6),
            sent_payload=sent,
        )


def make_multimodal_provider() -> OptionalMultimodalProvider:
    from ovascope_ml.providers.multimodal import DisabledMultimodalProvider

    if os.environ.get("OVASCOPE_MULTIMODAL_PROVIDER") == "meta_model_api":
        return MetaModelApiNarrator.from_environment()
    return DisabledMultimodalProvider()
