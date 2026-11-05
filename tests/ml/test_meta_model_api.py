"""Meta Model API narrator: minimisation, consent, validation, budget. No network is used."""

import io
import json

import pytest

from conftest import geometric_png, wait_job
from ovascope_ml.providers import meta_model_api as mm
from ovascope_ml.providers.base import ProviderError

RESULT = {
    "study_id": "a" * 32,
    "input_sha256": "b" * 64,
    "created_at": "2026-11-04T00:00:00Z",
    "status": "completed",
    "abstention_status": "not_abstained",
    "classification": {
        "pathway": "dinov2_whole",
        "highest_scoring_display": "Teratoma",
        "scores": [{"display": "Teratoma", "score": 0.61234}, {"display": "Simple cyst", "score": 0.2}],
    },
    "segmentation": {"status": "succeeded"},
    "quality_checks": [{"status": "warn", "message": "Frame size is outside the development range."}],
}


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen(content: str, captured: dict):
    def _open(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode())
        payload = {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
        }
        return FakeResp(json.dumps(payload).encode())

    return _open


def narrator(tmp_path, **kw):
    opts = {
        "api_key": "LLM_test",
        "transmission_authorised": True,
        "monthly_budget_usd": 1.0,
        "usage_path": tmp_path / "usage.json",
        **kw,
    }
    return mm.MetaModelApiNarrator(**opts)


def test_minimised_payload_excludes_identifiers_and_images():
    sent = mm.minimize(RESULT, {"balanced_accuracy": {"point": 0.51234}})
    blob = json.dumps(sent)
    for leak in ("a" * 32, "b" * 64, "2026-11-04", "study_id", "input_sha256", "data:image", "base64"):
        assert leak not in blob
    assert not any("image" in k or "artifact" in k for k in sent)  # no image or overlay fields at all
    assert sent["model_scores"][0]["score"] == 0.61 and sent["held_out_balanced_accuracy"] == 0.51


@pytest.mark.parametrize(
    "kw,reason",
    [
        ({"api_key": None}, "MODEL_API_KEY"),
        ({"transmission_authorised": False}, "not authorised"),
        ({"monthly_budget_usd": 0.0}, "budget"),
        ({"model": "muse-spark-1.3-contributor"}, "Contributor"),
        ({"base_url": "http://api.meta.ai/v1"}, "HTTPS"),
    ],
)
def test_unavailable_states_are_explicit(tmp_path, kw, reason):
    n = narrator(tmp_path, **kw)
    st = n.status()
    assert not st.available and reason in st.reason_unavailable
    with pytest.raises(ProviderError):
        n.narrate(mm.minimize(RESULT, None))


def test_valid_output_is_used_and_request_matches_documented_interface(tmp_path, monkeypatch):
    captured: dict = {}
    good = json.dumps(
        {
            "summary": "The research prototype's highest-scoring MMOTU category was Teratoma with a model score of 0.61; "
            "evaluation is image level without external validation.",
            "limitations": ["Single dataset."],
        }
    )
    monkeypatch.setattr(mm.urllib.request, "urlopen", fake_urlopen(good, captured))
    n = narrator(tmp_path)
    out = n.narrate(mm.minimize(RESULT, None))
    assert out.source == "meta-model-api" and out.validated and out.model_id == "muse-spark-1.3"
    assert captured["url"] == "https://api.meta.ai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer LLM_test"
    assert set(captured["body"]) == {"model", "messages", "max_completion_tokens", "temperature"}
    assert "a" * 32 not in json.dumps(captured["body"])
    assert out.estimated_cost_usd == pytest.approx(1000 / 1e6 * 1.25 + 200 / 1e6 * 4.25)
    assert n.budget.spent() == pytest.approx(out.estimated_cost_usd)


@pytest.mark.parametrize(
    "content,why",
    [
        ("not json", "valid JSON"),
        (
            json.dumps(
                {"summary": "This lesion is likely cancer according to the model output today.", "limitations": []}
            ),
            "forbidden",
        ),
        (
            json.dumps(
                {
                    "summary": "Teratoma scored 0.97 which is much higher than the other categories here.",
                    "limitations": [],
                }
            ),
            "number",
        ),
        (
            json.dumps({"summary": "A perfectly neutral summary of the supplied structured fields only.", "extra": 1}),
            "schema",
        ),
    ],
)
def test_unsafe_or_inventive_output_falls_back_to_template(tmp_path, monkeypatch, content, why):
    monkeypatch.setattr(mm.urllib.request, "urlopen", fake_urlopen(content, {}))
    out = narrator(tmp_path).narrate(mm.minimize(RESULT, None))
    assert out.source == "template" and not out.validated and why.lower() in out.fallback_reason.lower()
    assert "Research prototype" in out.text


def test_api_requires_consent_and_labels_narrative(api_factory, synthetic_engine, tmp_path, monkeypatch):
    eng = synthetic_engine()
    good = json.dumps(
        {
            "summary": "The research prototype produced model scores for the eight MMOTU categories; evaluation is image level.",
            "limitations": ["External validation not performed."],
        }
    )
    monkeypatch.setattr(mm.urllib.request, "urlopen", fake_urlopen(good, {}))
    client, _ = api_factory(eng)
    with client:
        sid = client.post("/api/v1/studies", files={"file": ("x.png", geometric_png(), "image/png")}).json()["id"]
        wait_job(client, client.post(f"/api/v1/studies/{sid}/analysis").json()["id"])
        # disabled provider -> 409, no consent -> 400
        assert (
            client.post(f"/api/v1/studies/{sid}/narrative", json={"consent_external_transmission": True}).status_code
            == 409
        )
        eng.multimodal = narrator(tmp_path)
        assert client.post(f"/api/v1/studies/{sid}/narrative", json={}).json()["code"] == "CONSENT_REQUIRED"
        prev = client.get(f"/api/v1/studies/{sid}/narrative/preview").json()
        assert prev["provider"]["available"] and "study_id" not in json.dumps(prev["would_send"])
        r = client.post(f"/api/v1/studies/{sid}/narrative", json={"consent_external_transmission": True})
        assert r.status_code == 201 and r.json()["source"] == "meta-model-api"
        md = client.get(f"/api/v1/studies/{sid}/report?format=md").text
        assert "Narrative summary" in md and "muse-spark-1.3 via Meta Model API" in md
        caps = client.get("/api/v1/capabilities").json()
        assert caps["external_transmission"]["allowed"] is False  # images never leave
        assert caps["external_transmission"]["structured_results_to_meta_model_api"] is True


def test_billing_error_is_typed_and_redacted(tmp_path, monkeypatch):
    import urllib.error

    def raise_402(req, timeout=None):
        body = io.BytesIO(
            json.dumps(
                {"error": {"code": "billing_not_configured", "message": "Billing verification failed for LLM_test."}}
            ).encode()
        )
        raise urllib.error.HTTPError(req.full_url, 402, "Payment Required", {}, body)

    monkeypatch.setattr(mm.urllib.request, "urlopen", raise_402)
    n = narrator(tmp_path)
    with pytest.raises(ProviderError) as e:
        n.narrate(mm.minimize(RESULT, None))
    assert e.value.code == "PROVIDER_BILLING" and "Billing verification failed" in str(e.value)
    assert "LLM_test" not in str(e.value)
    assert n.budget.spent() == 0.0  # failed calls are not charged to the budget
