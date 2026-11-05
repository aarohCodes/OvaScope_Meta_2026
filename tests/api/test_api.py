"""API contract, security and lifecycle tests (synthetic geometric images only)."""

import struct
import zlib

import pytest

from conftest import geometric_png, wait_job


def _upload(client, data=None, name="image.png", ctype="image/png"):
    return client.post("/api/v1/studies", files={"file": (name, data or geometric_png(), ctype)})


def test_health_and_capabilities_in_no_model_mode(api_factory, no_model_engine):
    client, _ = api_factory(no_model_engine)
    with client:
        h = client.get("/health").json()
        assert h["mode"] == "NO_MODEL" and "Research prototype" in h["notice"]
        c = client.get("/api/v1/capabilities").json()
        assert c["mode"] == "NO_MODEL" and c["active_bundle"] is None
        assert c["external_transmission"]["allowed"] is False
        assert {p["kind"] for p in c["providers"]} >= {
            "representation",
            "segmentation",
            "classifier",
            "multimodal",
            "artifact_store",
        }
        assert any(m["component"] == "classifier" for m in c["missing_requirements"])


def test_upload_validation_and_typed_errors(api_factory, no_model_engine):
    client, _ = api_factory(no_model_engine, max_upload_bytes=200_000)
    with client:
        r = _upload(client)
        assert r.status_code == 201 and r.json()["width"] == 400
        assert "image.png" not in r.text  # file names are not stored or echoed
        r = _upload(client, b"GIF89a" + b"0" * 64, "x.gif", "image/gif")
        assert r.status_code == 415 and r.json()["code"] == "UNSUPPORTED_FORMAT" and r.json()["request_id"]
        r = _upload(client, b"\x89PNG\r\n\x1a\n" + b"garbage" * 10)
        assert r.status_code == 422 and r.json()["code"] == "CORRUPT_IMAGE"
        r = _upload(client, b"\x89PNG\r\n\x1a\n" + b"0" * 300_000)
        assert r.status_code == 413 and r.json()["code"] == "FILE_TOO_LARGE"
        # disguised file: PNG extension and content type, JPEG content -> accepted as JPEG by sniffing
        r = _upload(client, geometric_png(fmt="JPEG"), "x.png", "image/png")
        assert r.status_code == 201 and r.json()["image_format"] == "JPEG"


def test_decompression_bomb_rejected(api_factory, no_model_engine):
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 50_000, 50_000, 8, 2, 0, 0, 0)
    chunk = struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))
    client, _ = api_factory(no_model_engine)
    with client:
        r = _upload(client, sig + chunk + iend)
        assert r.status_code == 413 and r.json()["code"] == "DECOMPRESSION_BOMB"


def test_no_model_analysis_returns_no_score(api_factory, no_model_engine):
    client, _ = api_factory(no_model_engine)
    with client:
        sid = _upload(client).json()["id"]
        job = client.post(f"/api/v1/studies/{sid}/analysis").json()
        assert wait_job(client, job["id"])["status"] == "succeeded"
        res = client.get(f"/api/v1/studies/{sid}/results").json()["results"][0]
        assert res["status"] == "model_unavailable"
        assert res["model_score_or_null"] is None and res["classification"] is None
        assert res["segmentation_artifact_or_null"] is None
        md = client.get(f"/api/v1/studies/{sid}/report?format=md").text
        assert "Model unavailable" in md and "Research prototype" in md
        for banned in ("probability of", "cancer-free", "you have cancer", "risk of cancer"):
            assert banned not in md.lower()


def test_full_inference_flow_artifacts_and_deletion(api_factory, synthetic_engine, tmp_path):
    client, app = api_factory(synthetic_engine())
    with client:
        sid = _upload(client).json()["id"]
        job = client.post(f"/api/v1/studies/{sid}/analysis", headers={"Idempotency-Key": "k-123456"}).json()
        assert wait_job(client, job["id"])["status"] == "succeeded"
        res = client.get(f"/api/v1/studies/{sid}/results").json()["results"][0]
        assert res["status"] in ("completed", "completed_with_warnings")
        assert res["calibration_status"] == "temperature_scaled_not_prevalence_calibrated"
        art = res["segmentation_artifact_or_null"]
        assert art and art["width"] == 400 and art["height"] == 300 and "/" not in art["artifact_id"]
        png = client.get(f"/api/v1/studies/{sid}/artifacts/{art['artifact_id']}")
        assert png.status_code == 200 and png.headers["content-type"] == "image/png"
        assert "no-store" in png.headers["cache-control"]
        html = client.get(f"/api/v1/studies/{sid}/report?format=html")
        assert (
            html.status_code == 200
            and "Research prototype" in html.text
            and "attachment" in html.headers["content-disposition"]
        )
        # idempotency: same key -> same job; no key after success -> reused; force -> new job
        again = client.post(f"/api/v1/studies/{sid}/analysis", headers={"Idempotency-Key": "k-123456"})
        assert again.status_code == 200 and again.json()["id"] == job["id"] and again.json()["reused"]
        assert client.post(f"/api/v1/studies/{sid}/analysis").json()["reused"]
        forced = client.post(f"/api/v1/studies/{sid}/analysis?force=true").json()
        assert forced["id"] != job["id"]
        wait_job(client, forced["id"])
        store_dir = tmp_path / "store" / sid
        assert store_dir.exists() and any(store_dir.iterdir())
        rec = client.delete(f"/api/v1/studies/{sid}").json()
        assert rec["deleted"] and rec["artifacts_removed"] >= 4 and rec["results_removed"] == 2
        assert not store_dir.exists()
        assert client.get(f"/api/v1/studies/{sid}").status_code == 404
        assert client.get(f"/api/v1/studies/{sid}/artifacts/{art['artifact_id']}").status_code == 404


def test_provider_timeout_is_retried_then_reported(api_factory, synthetic_engine):
    eng = synthetic_engine()
    from ovascope_ml.inference.pipeline import EngineOutput
    from ovascope_ml.inference.result_schema import AnalysisResult, ErrorInfo, ResultStatus

    calls = {"n": 0}

    def timeout_analyze(data, study_id, cancel_check=None, on_stage=None):
        calls["n"] += 1
        r = AnalysisResult(
            study_id=study_id,
            status=ResultStatus.failed,
            input_sha256="x",
            error=ErrorInfo(code="PROVIDER_TIMEOUT", message="timed out", retryable=True),
        )
        return EngineOutput(r, [])

    eng.analyze = timeout_analyze  # type: ignore[method-assign]
    client, _ = api_factory(eng, job_max_attempts=2)
    with client:
        sid = _upload(client).json()["id"]
        j = wait_job(client, client.post(f"/api/v1/studies/{sid}/analysis").json()["id"])
        assert j["status"] == "failed" and j["attempts"] == 2 and j["error_code"] == "PROVIDER_TIMEOUT"
        assert calls["n"] == 2
        res = client.get(f"/api/v1/studies/{sid}/results").json()["results"][0]
        assert res["status"] == "failed" and res["model_score_or_null"] is None


def test_worker_crash_is_bounded_and_explicit(api_factory, synthetic_engine):
    eng = synthetic_engine()

    def crash(*a, **k):
        raise RuntimeError("boom")

    eng.analyze = crash  # type: ignore[method-assign]
    client, _ = api_factory(eng, job_max_attempts=2)
    with client:
        sid = _upload(client).json()["id"]
        j = wait_job(client, client.post(f"/api/v1/studies/{sid}/analysis").json()["id"])
        assert j["status"] == "failed" and j["error_code"] == "WORKER_ERROR" and j["attempts"] == 2
        assert client.get(f"/api/v1/studies/{sid}/results").json()["results"] == []


def test_cancel_queued_job(api_factory, no_model_engine):
    client, _ = api_factory(no_model_engine, inline_worker=False)
    with client:
        sid = _upload(client).json()["id"]
        job = client.post(f"/api/v1/studies/{sid}/analysis").json()
        c = client.post(f"/api/v1/jobs/{job['id']}/cancel").json()
        assert c["status"] == "cancelled"


def test_queue_limit(api_factory, no_model_engine):
    client, _ = api_factory(no_model_engine, inline_worker=False, max_queued_jobs=1)
    with client:
        a, b = _upload(client).json()["id"], _upload(client).json()["id"]
        assert client.post(f"/api/v1/studies/{a}/analysis").status_code == 202
        r = client.post(f"/api/v1/studies/{b}/analysis")
        assert r.status_code == 429 and r.json()["code"] == "QUEUE_FULL"


def test_annotations_are_bounded_and_labelled_manual(api_factory, no_model_engine):
    client, _ = api_factory(no_model_engine)
    with client:
        sid = _upload(client).json()["id"]
        ok = client.post(f"/api/v1/studies/{sid}/annotations", json={"polygon": [[10, 10], [100, 10], [50, 80]]})
        assert ok.status_code == 201 and ok.json()["used_by_automatic_pipeline"] is False
        rep = client.get(f"/api/v1/studies/{sid}/report?format=json").json()
        assert (
            len(rep["researcher_annotations"]["items"]) == 1
            and "not model output" in rep["researcher_annotations"]["note"]
        )
        bad = client.post(f"/api/v1/studies/{sid}/annotations", json={"polygon": [[10, 10], [9999, 10], [50, 80]]})
        assert bad.status_code == 422 and bad.json()["code"] == "ANNOTATION_OUT_OF_BOUNDS"


def test_path_traversal_and_malformed_ids(api_factory, no_model_engine):
    client, _ = api_factory(no_model_engine)
    with client:
        for path in (
            "/api/v1/studies/..%2F..%2Fetc",
            "/api/v1/studies/abc",
            "/api/v1/jobs/../../x",
            "/api/v1/evaluation/cases/..%2Fsecret/image",
        ):
            assert client.get(path).status_code in (404, 422)


def test_evaluation_endpoint_without_artifacts_is_honest(api_factory, no_model_engine, monkeypatch, tmp_path):
    monkeypatch.setenv("OVASCOPE_EVALUATION_DIR", str(tmp_path / "noeval"))
    client, _ = api_factory(no_model_engine)
    with client:
        ev = client.get("/api/v1/evaluation").json()
        assert ev["evaluation"]["available"] is False and "Run" in ev["evaluation"]["reason"]
        assert ev["data_audit"] is None


def test_public_deployment_requires_token_and_auth_is_enforced(api_factory, no_model_engine):
    with pytest.raises(RuntimeError, match="requires OVASCOPE_API_TOKEN"):
        api_factory(no_model_engine, deployment="public")
    client, _ = api_factory(no_model_engine, deployment="public", api_token="t" * 32)
    with client:
        assert client.get("/api/v1/capabilities").status_code == 401
        assert client.get("/api/v1/capabilities", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/api/v1/capabilities", headers={"Authorization": "Bearer " + "t" * 32}).status_code == 200
        assert client.get("/docs").status_code == 404  # no interactive docs in public mode


def test_cors_is_restricted(api_factory, no_model_engine):
    client, _ = api_factory(no_model_engine)
    with client:
        evil = client.options(
            "/api/v1/capabilities", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"}
        )
        assert "access-control-allow-origin" not in {k.lower() for k in evil.headers}
        ok = client.options(
            "/api/v1/capabilities", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"}
        )
        assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_no_phi_in_logs(api_factory, no_model_engine, capsys):
    client, _ = api_factory(no_model_engine)
    with client:
        sid = _upload(client, name="Jane_Doe_MRN-0042_2026-01-01.png").json()["id"]
        wait_job(
            client,
            client.post(
                f"/api/v1/studies/{sid}/analysis", headers={"Authorization": "Bearer secret-token-value"}
            ).json()["id"],
        )
    out = capsys.readouterr()
    logs = out.out + out.err
    assert "study created" in logs  # structured logs are emitted...
    for leak in ("Jane", "MRN-0042", "secret-token-value", "\x89PNG"):
        assert leak not in logs  # ...without file names, tokens or image bytes


def test_security_headers_present(api_factory, no_model_engine):
    client, _ = api_factory(no_model_engine)
    with client:
        r = client.get("/api/v1/capabilities")
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert "no-store" in r.headers["cache-control"]
        assert len(r.headers["x-request-id"]) >= 8


def test_env_file_loader_is_allow_listed_and_never_overrides(tmp_path, monkeypatch):
    from ovascope_api.settings import load_env_file

    env = tmp_path / ".env"
    env.write_text(
        "# comment\nMODEL_API_KEY=LLM_secret\nOVASCOPE_META_MODEL='muse-spark-1.3'\n"
        "OVASCOPE_RETENTION_HOURS=5\nAWS_SECRET_ACCESS_KEY=nope\nexport OVASCOPE_INFERENCE_MODE=no_model\n",
        encoding="utf-8",
    )
    for k in ("MODEL_API_KEY", "OVASCOPE_META_MODEL", "OVASCOPE_INFERENCE_MODE", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OVASCOPE_RETENTION_HOURS", "24")
    loaded = load_env_file(env)
    import os

    assert set(loaded) == {"MODEL_API_KEY", "OVASCOPE_META_MODEL", "OVASCOPE_INFERENCE_MODE"}
    assert os.environ["OVASCOPE_META_MODEL"] == "muse-spark-1.3"
    assert os.environ["OVASCOPE_RETENTION_HOURS"] == "24"  # existing value wins
    assert "AWS_SECRET_ACCESS_KEY" not in os.environ  # not allow-listed
    for k in loaded:
        monkeypatch.delenv(k, raising=False)


def test_comma_separated_cors_from_env_file(tmp_path):
    from ovascope_api.settings import Settings

    env = tmp_path / ".env"
    env.write_text("OVASCOPE_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000\n", encoding="utf-8")
    s = Settings(_env_file=str(env))
    assert s.cors_origins == ["http://localhost:3000", "http://127.0.0.1:3000"]
