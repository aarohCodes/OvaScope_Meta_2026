"""OvaScope API (FastAPI). Research prototype. Not for diagnosis, screening, or treatment decisions."""

from __future__ import annotations

import collections
import hashlib
import hmac
import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from sqlalchemy import delete, func, select, update

from ovascope_api import jobs as jobs_mod
from ovascope_api.db import Annotation, Artifact, Job, Result, Study, make_engine, make_session_factory
from ovascope_api.logging_setup import configure_logging, request_id_var
from ovascope_api.report import build_report, render_html, render_markdown
from ovascope_api.schemas import (
    AnnotationIn,
    AnnotationOut,
    CapabilitiesOut,
    DeletionReceipt,
    JobOut,
    ProviderOut,
    StudyOut,
)
from ovascope_api.settings import Settings, get_settings
from ovascope_ml import RESEARCH_NOTICE, paths
from ovascope_ml.data.imaging import ImageInputError, ImageLimits, decode_image
from ovascope_ml.inference.pipeline import AnalysisEngine
from ovascope_ml.providers.artifact_store import InvalidArtifactKey, LocalArtifactStore

log = logging.getLogger("ovascope.api")
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
Mode = Literal["RESEARCH_INFERENCE", "NO_MODEL"]
_ID = re.compile(r"^[a-f0-9]{32}$")
_IMG_ID = re.compile(r"^mmotu2d-[0-9]{1,6}$")
_REQ_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
MEDIA = {"PNG": "image/png", "JPEG": "image/jpeg"}


def api_error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.hits: dict[str, collections.deque[float]] = collections.defaultdict(collections.deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True


class BodySizeLimit:
    """Pure ASGI middleware: abort uploads exceeding the limit before multipart parsing."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        cl = headers.get(b"content-length")
        if cl is not None and cl.isdigit() and int(cl) > self.max_bytes:
            return await _reject_413(send)
        total = 0

        async def limited_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    raise _TooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _TooLarge:
            await _reject_413(send)


class _TooLarge(Exception):
    pass


async def _reject_413(send) -> None:
    body = json.dumps({"code": "FILE_TOO_LARGE", "message": "Upload exceeds the size limit."}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})


def create_app(settings: Settings | None = None, engine: AnalysisEngine | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.assert_safe()
    configure_logging()
    db_engine = make_engine(settings.database_url)
    sf = make_session_factory(db_engine)
    if settings.auto_migrate:
        from ovascope_api.migrate import upgrade

        upgrade(settings.database_url)
    store = LocalArtifactStore(Path(settings.artifact_dir))
    limiter = RateLimiter(settings.uploads_per_minute)
    limits = ImageLimits(max_bytes=settings.max_upload_bytes, max_pixels=settings.max_image_pixels)
    state: dict[str, Any] = {"engine": engine}
    engine_lock = threading.Lock()

    def get_engine() -> AnalysisEngine:
        with engine_lock:
            if state["engine"] is None:
                state["engine"] = AnalysisEngine.from_environment()
            return state["engine"]

    worker = jobs_mod.WorkerLoop(sf, store, get_engine, inline=True) if settings.inline_worker else None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        jobs_mod.purge_expired(sf, store)
        if worker:
            worker.start_thread()
        yield
        if worker:
            worker.stop()

    app = FastAPI(
        title="OvaScope API",
        version="0.1.0",
        lifespan=lifespan,
        description=RESEARCH_NOTICE,
        docs_url="/docs" if settings.deployment == "local" else None,
        redoc_url=None,
    )
    app.state.sf, app.state.store, app.state.settings, app.state.get_engine = sf, store, settings, get_engine
    app.state.worker = worker

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization", "Idempotency-Key", "X-Request-ID"],
        allow_credentials=False,
        max_age=600,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id")
        rid = rid if rid and _REQ_ID.match(rid) else uuid.uuid4().hex
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers.setdefault("Cache-Control", "no-store")
        if not request.url.path.startswith("/docs") and request.url.path != "/openapi.json":
            response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        log.info(
            "request",
            extra={
                "method": request.method,
                "path": _route_template(request),
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        )
        return response

    app.add_middleware(BodySizeLimit, max_bytes=settings.max_upload_bytes + 64 * 1024)

    @app.exception_handler(HTTPException)
    async def http_exc(_req: Request, exc: HTTPException):
        d = exc.detail if isinstance(exc.detail, dict) else {"code": "HTTP_ERROR", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={**d, "request_id": request_id_var.get()})

    @app.exception_handler(RequestValidationError)
    async def val_exc(_req: Request, exc: RequestValidationError):
        fields = [".".join(str(p) for p in e.get("loc", [])) for e in exc.errors()]
        return JSONResponse(
            status_code=422,
            content={
                "code": "VALIDATION_ERROR",
                "message": f"Invalid request fields: {', '.join(fields)}",
                "request_id": request_id_var.get(),
            },
        )

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if settings.api_token is None:
            return
        expected = f"Bearer {settings.api_token.get_secret_value()}"
        if authorization is None or not hmac.compare_digest(authorization.encode(), expected.encode()):
            raise api_error(401, "UNAUTHORIZED", "Missing or invalid credentials.")

    v1 = APIRouter(prefix="/api/v1", dependencies=[Depends(require_auth)])

    def _study(s, study_id: str) -> Study:
        if not _ID.match(study_id):
            raise api_error(404, "STUDY_NOT_FOUND", "Study not found.")
        st = s.get(Study, study_id)
        if st is None:
            raise api_error(404, "STUDY_NOT_FOUND", "Study not found.")
        return st

    def _study_out(s, st: Study) -> StudyOut:
        latest = s.scalars(select(Job.id).where(Job.study_id == st.id).order_by(Job.created_at.desc()).limit(1)).first()
        n = s.scalar(select(func.count()).select_from(Result).where(Result.study_id == st.id)) or 0
        return StudyOut(
            id=st.id,
            created_at=st.created_at,
            retention_expires_at=st.retention_expires_at,
            input_sha256=st.input_sha256,
            image_format=st.image_format,
            width=st.width,
            height=st.height,
            size_bytes=st.size_bytes,
            external_transmission_consent=st.external_transmission_consent,
            latest_job_id=latest,
            n_results=n,
        )

    def _job_out(j: Job, reused: bool = False) -> JobOut:
        return JobOut(
            id=j.id,
            study_id=j.study_id,
            status=cast(JobStatus, j.status),
            stage=j.stage,
            attempts=j.attempts,
            max_attempts=j.max_attempts,
            error_code=j.error_code,
            error_message=j.error_message,
            created_at=j.created_at,
            started_at=j.started_at,
            finished_at=j.finished_at,
            reused=reused,
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        db_ok = True
        try:
            with sf() as s:
                s.execute(select(1))
                hb = jobs_mod.latest_heartbeat(s)
        except Exception:
            db_ok, hb = False, None
        age = None if hb is None else (datetime.now(UTC) - _aware(hb.last_seen)).total_seconds()
        return {
            "status": "ok" if db_ok else "degraded",
            "database": "ok" if db_ok else "unavailable",
            "worker": {
                "online": age is not None and age < 30,
                "last_seen_seconds_ago": age,
                "inline": bool(hb.inline) if hb else None,
            },
            "mode": get_engine().mode(),
            "notice": RESEARCH_NOTICE,
        }

    @v1.get("/capabilities", response_model=CapabilitiesOut)
    def capabilities() -> CapabilitiesOut:
        eng = get_engine()
        with sf() as s:
            hb = jobs_mod.latest_heartbeat(s)
        age = None if hb is None else (datetime.now(UTC) - _aware(hb.last_seen)).total_seconds()
        provs = [ProviderOut(**{k: v for k, v in st.__dict__.items()}) for st in eng.statuses()]
        provs.append(ProviderOut(**store.status().__dict__))
        mm = eng.multimodal.status()
        b = eng.bundle.meta if (eng.bundle and eng.mode() == "RESEARCH_INFERENCE") else None
        return CapabilitiesOut(
            notice=RESEARCH_NOTICE,
            mode=cast(Mode, eng.mode()),
            deployment=settings.deployment,
            providers=provs,
            missing_requirements=[m.model_dump() for m in eng.missing_requirements()],
            worker={
                "online": age is not None and age < 30,
                "last_seen_seconds_ago": age,
                "inline": bool(hb.inline) if hb else None,
                "concurrency": 1,
            },
            limits={
                "max_upload_bytes": settings.max_upload_bytes,
                "max_image_pixels": settings.max_image_pixels,
                "min_side_px": limits.min_side,
                "max_side_px": limits.max_side,
                "uploads_per_minute": settings.uploads_per_minute,
                "max_queued_jobs": settings.max_queued_jobs,
                "job_max_attempts": settings.job_max_attempts,
                "video_frames": "not supported in this version",
            },
            accepted_formats=["image/png", "image/jpeg"],
            external_transmission={
                "allowed": False,
                "policy": "fail closed",
                "reason": "Images never leave this server; all model inference runs locally.",
                "structured_results_to_meta_model_api": mm.available,
                "structured_results_detail": (
                    "Optional narrative text: minimised structured result fields (no image, IDs or hashes) "
                    "may be sent to the Meta Model API with per-call consent."
                    if mm.available
                    else "Disabled: " + (mm.reason_unavailable or "not configured")
                ),
            },
            retention={
                "hours": settings.retention_hours,
                "deletion": "DELETE /api/v1/studies/{id} removes the study, "
                "its jobs, results, annotations and stored artifacts immediately",
            },
            budget={
                "hosted_provider_monthly_cap_usd": _num(mm.details.get("monthly_budget_usd")),
                "hosted_provider_spent_this_month_usd": _num(mm.details.get("spent_this_month_usd")),
                "hosted_calls_enabled": mm.available,
            },
            active_bundle=None
            if b is None
            else {
                "bundle_id": b["bundle_id"],
                "model_id": b["model_id"],
                "model_version": b["model_version"],
                "checkpoint_sha256": b["checkpoint_sha256"],
                "primary_pathway": b["primary_pathway"],
                "classification_target": b["classification_target"],
                "calibration_status": b["calibration"]["status"],
            },
        )

    @v1.post("/studies", status_code=201, response_model=StudyOut)
    async def create_study(
        request: Request, file: UploadFile = File(...), external_transmission_consent: bool = Form(False)
    ) -> StudyOut:
        client = request.client.host if request.client else "unknown"
        if not limiter.check(client):
            raise api_error(429, "RATE_LIMITED", "Too many uploads; try again in a minute.")
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise api_error(413, "FILE_TOO_LARGE", "Upload exceeds the size limit.")
        try:
            img = decode_image(data, limits)
        except ImageInputError as exc:
            status = (
                413
                if exc.code.value in ("FILE_TOO_LARGE", "DECOMPRESSION_BOMB")
                else 415
                if exc.code.value == "UNSUPPORTED_FORMAT"
                else 422
            )
            raise api_error(status, exc.code.value, exc.message) from None
        with sf() as s:
            st = Study(
                retention_expires_at=datetime.now(UTC) + timedelta(hours=settings.retention_hours),
                input_sha256=hashlib.sha256(data).hexdigest(),
                image_format=img.format,
                width=img.width,
                height=img.height,
                size_bytes=len(data),
                image_key="pending",
                external_transmission_consent=bool(external_transmission_consent),
            )
            s.add(st)
            s.flush()
            st.image_key = store.put(st.id, data, MEDIA[img.format])
            s.commit()
            log.info("study created", extra={"study_id": st.id})
            return _study_out(s, st)

    @v1.get("/studies/{study_id}", response_model=StudyOut)
    def get_study(study_id: str) -> StudyOut:
        with sf() as s:
            return _study_out(s, _study(s, study_id))

    @v1.get("/studies/{study_id}/image")
    def get_image(study_id: str) -> Response:
        with sf() as s:
            st = _study(s, study_id)
            return Response(
                store.get(st.image_key),
                media_type=MEDIA[st.image_format],
                headers={"Cache-Control": "private, no-store", "Content-Disposition": "inline"},
            )

    @v1.post("/studies/{study_id}/analysis", status_code=202, response_model=JobOut)
    def request_analysis(
        study_id: str,
        response: Response,
        force: bool = Query(False),
        idempotency_key: str | None = Header(default=None, max_length=128),
    ) -> JobOut:
        eng = get_engine()
        fp = jobs_mod.engine_fingerprint(eng)
        with sf() as s:
            _study(s, study_id)
            if idempotency_key:
                existing = s.scalars(
                    select(Job).where(Job.study_id == study_id, Job.idempotency_key == idempotency_key)
                ).first()
                if existing:
                    response.status_code = 200
                    return _job_out(existing, reused=True)
            active = s.scalars(
                select(Job).where(Job.study_id == study_id, Job.status.in_(("queued", "running")))
            ).first()
            if active:
                response.status_code = 200
                return _job_out(active, reused=True)
            if not force:
                done = s.scalars(
                    select(Job)
                    .where(Job.study_id == study_id, Job.status == "succeeded", Job.engine_fingerprint == fp)
                    .order_by(Job.created_at.desc())
                ).first()
                if done:
                    response.status_code = 200
                    return _job_out(done, reused=True)
            queued = s.scalar(select(func.count()).select_from(Job).where(Job.status == "queued")) or 0
            if queued >= settings.max_queued_jobs:
                raise api_error(429, "QUEUE_FULL", "The analysis queue is full; try again later.")
            j = Job(
                study_id=study_id,
                status="queued",
                max_attempts=settings.job_max_attempts,
                idempotency_key=idempotency_key,
                engine_fingerprint=fp,
                request_id=request_id_var.get(),
            )
            s.add(j)
            s.commit()
            log.info("job queued", extra={"job_id": j.id, "study_id": study_id})
            return _job_out(j)

    @v1.get("/jobs/{job_id}", response_model=JobOut)
    def get_job(job_id: str) -> JobOut:
        with sf() as s:
            j = s.get(Job, job_id) if _ID.match(job_id) else None
            if j is None:
                raise api_error(404, "JOB_NOT_FOUND", "Job not found.")
            return _job_out(j)

    @v1.post("/jobs/{job_id}/cancel", response_model=JobOut)
    def cancel_job(job_id: str) -> JobOut:
        with sf() as s:
            j = s.get(Job, job_id) if _ID.match(job_id) else None
            if j is None:
                raise api_error(404, "JOB_NOT_FOUND", "Job not found.")
            if j.status == "queued":
                j.status, j.finished_at = "cancelled", datetime.now(UTC)
            elif j.status == "running":
                j.cancel_requested = True
            s.commit()
            return _job_out(j)

    @v1.get("/studies/{study_id}/results")
    def get_results(study_id: str) -> dict[str, Any]:
        with sf() as s:
            _study(s, study_id)
            rows = s.scalars(select(Result).where(Result.study_id == study_id).order_by(Result.created_at.desc())).all()
            return {"study_id": study_id, "results": [r.payload for r in rows]}

    @v1.get("/studies/{study_id}/artifacts/{artifact_id}")
    def get_artifact(study_id: str, artifact_id: str) -> Response:
        with sf() as s:
            _study(s, study_id)
            a = s.get(Artifact, artifact_id) if _ID.match(artifact_id) else None
            if a is None or a.study_id != study_id:
                raise api_error(404, "ARTIFACT_NOT_FOUND", "Artifact not found.")
            try:
                data = store.get(a.storage_key)
            except (InvalidArtifactKey, FileNotFoundError):
                raise api_error(404, "ARTIFACT_NOT_FOUND", "Artifact not found.") from None
            return Response(data, media_type=a.media_type, headers={"Cache-Control": "private, no-store"})

    @v1.post("/studies/{study_id}/annotations", status_code=201, response_model=AnnotationOut)
    def add_annotation(study_id: str, body: AnnotationIn) -> AnnotationOut:
        with sf() as s:
            st = _study(s, study_id)
            for x, y in ((p[0], p[1]) for p in body.polygon if len(p) == 2):
                if not (0 <= x <= st.width and 0 <= y <= st.height):
                    raise api_error(422, "ANNOTATION_OUT_OF_BOUNDS", "Polygon points must lie within the image.")
            if any(len(p) != 2 for p in body.polygon):
                raise api_error(422, "ANNOTATION_INVALID", "Polygon points must be [x, y] pairs.")
            a = Annotation(study_id=study_id, polygon=body.polygon, note=body.note)
            s.add(a)
            s.commit()
            return AnnotationOut(id=a.id, study_id=study_id, polygon=a.polygon, note=a.note, created_at=a.created_at)

    @v1.get("/studies/{study_id}/annotations")
    def list_annotations(study_id: str) -> dict[str, Any]:
        with sf() as s:
            _study(s, study_id)
            rows = s.scalars(
                select(Annotation).where(Annotation.study_id == study_id).order_by(Annotation.created_at)
            ).all()
            return {
                "study_id": study_id,
                "annotations": [
                    AnnotationOut(
                        id=a.id, study_id=study_id, polygon=a.polygon, note=a.note, created_at=a.created_at
                    ).model_dump(mode="json")
                    for a in rows
                ],
            }

    @v1.delete("/studies/{study_id}", response_model=DeletionReceipt)
    def delete_study(study_id: str) -> DeletionReceipt:
        with sf() as s:
            st = _study(s, study_id)
            counts = {
                "results_removed": len(st.results),
                "jobs_removed": len(st.jobs),
                "annotations_removed": len(st.annotations),
            }
            # Signal any running job, then let the database's ON DELETE CASCADE remove
            # jobs, results, artifacts and annotations in one statement.
            s.execute(update(Job).where(Job.study_id == study_id).values(cancel_requested=True))
            s.execute(delete(Study).where(Study.id == study_id))
            s.commit()
        removed = store.delete_namespace(study_id)
        log.info("study deleted", extra={"study_id": study_id})
        return DeletionReceipt(study_id=study_id, deleted=True, artifacts_removed=removed, **counts)

    @v1.get("/models")
    def models() -> dict[str, Any]:
        from ovascope_ml.models.registry import NOT_USED, REGISTRY

        eng = get_engine()
        b = eng.bundle.meta if eng.bundle else None
        return {
            "mode": eng.mode(),
            "active_bundle": b,
            "bundle_error": eng.bundle_error if b is None else None,
            "third_party_weights": [
                {
                    "key": w.key,
                    "repo_id": w.repo_id,
                    "revision": w.revision,
                    "license": w.license,
                    "publisher": w.publisher,
                    "meta_model": w.meta_model,
                    "role": w.role,
                }
                for w in REGISTRY.values()
            ],
            "not_used": NOT_USED,
            "comparators": [
                {"id": "torchvision/resnet50.IMAGENET1K_V2", "role": "evaluation comparator only (not deployed)"}
            ],
        }

    @v1.get("/evaluation")
    def evaluation() -> dict[str, Any]:
        d = paths.evaluation_dir()
        out: dict[str, Any] = {"notice": RESEARCH_NOTICE}
        for name in ("evaluation", "data_audit", "mirror_verification"):
            p = d / f"{name}.json"
            out[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        if out["evaluation"] is None:
            out["evaluation"] = {
                "available": False,
                "reason": "No evaluation artifact found. Run `make evaluate` after training on an authorised dataset.",
            }
        return out

    @v1.get("/evaluation/cases/{image_id}/image")
    def evaluation_case_image(image_id: str) -> Response:
        if not settings.serve_eval_images:
            raise api_error(404, "DISABLED", "Serving evaluation images is disabled (OVASCOPE_SERVE_EVAL_IMAGES).")
        if not _IMG_ID.match(image_id):
            raise api_error(404, "NOT_FOUND", "Case not found.")
        p = paths.evaluation_dir() / "evaluation.json"
        ev = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        allowed = {
            c["image_id"]
            for k in ("confident_errors", "abstained", "hgs_below_threshold")
            for c in ev.get("failure_cases", {}).get(k, [])
        }
        if image_id not in allowed:
            raise api_error(404, "NOT_FOUND", "Only listed failure cases can be viewed.")
        f = paths.mmotu_root() / "OTU_2d" / "images" / f"{image_id.removeprefix('mmotu2d-')}.JPG"
        if not f.exists():
            raise api_error(404, "NOT_FOUND", "Dataset image not present locally.")
        return Response(f.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})

    @v1.get("/studies/{study_id}/report")
    def report(study_id: str, format: str = Query("json", pattern="^(json|md|html)$")) -> Response:
        with sf() as s:
            st = _study(s, study_id)
            so = _study_out(s, st).model_dump(mode="json")
            latest = s.scalars(
                select(Result).where(Result.study_id == study_id).order_by(Result.created_at.desc())
            ).first()
            anns = [
                {"id": a.id, "polygon": a.polygon, "note": a.note, "created_at": a.created_at.isoformat()}
                for a in s.scalars(select(Annotation).where(Annotation.study_id == study_id)).all()
            ]
            narr_row = s.scalars(
                select(Artifact)
                .where(Artifact.study_id == study_id, Artifact.kind == "narrative")
                .order_by(Artifact.created_at.desc())
            ).first()
            narrative = json.loads(store.get(narr_row.storage_key)) if narr_row else None
        ev_path = paths.evaluation_dir() / "evaluation.json"
        ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None
        rep = build_report(so, latest.payload if latest else None, ev, anns)
        rep["narrative"] = narrative
        fname = f"ovascope-report-{study_id[:8]}"
        if format == "md":
            return PlainTextResponse(
                render_markdown(rep),
                media_type="text/markdown",
                headers={"Content-Disposition": f'attachment; filename="{fname}.md"'},
            )
        if format == "html":
            return HTMLResponse(
                render_html(rep),
                headers={
                    "Content-Disposition": f'attachment; filename="{fname}.html"',
                    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
                },
            )
        return JSONResponse(rep)

    def _narrative_inputs(study_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        with sf() as s:
            _study(s, study_id)
            latest = s.scalars(
                select(Result).where(Result.study_id == study_id).order_by(Result.created_at.desc())
            ).first()
        if latest is None:
            raise api_error(409, "NO_RESULT", "Run an analysis before requesting a narrative.")
        ev_path = paths.evaluation_dir() / "evaluation.json"
        ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None
        rep = build_report({"id": study_id}, latest.payload, ev)
        return latest.payload, rep.get("evaluation_summary")

    @v1.get("/studies/{study_id}/narrative/preview")
    def narrative_preview(study_id: str) -> dict[str, Any]:
        from ovascope_ml.providers.meta_model_api import minimize

        payload, ev_summary = _narrative_inputs(study_id)
        st = get_engine().multimodal.status()
        return {"provider": ProviderOut(**st.__dict__).model_dump(), "would_send": minimize(payload, ev_summary)}

    @v1.post("/studies/{study_id}/narrative", status_code=201)
    def create_narrative(study_id: str, body: dict[str, Any]) -> dict[str, Any]:
        from dataclasses import asdict

        from ovascope_ml.providers.base import ProviderError
        from ovascope_ml.providers.meta_model_api import MetaModelApiNarrator, minimize

        if body.get("consent_external_transmission") is not True:
            raise api_error(
                400, "CONSENT_REQUIRED", "Explicit consent is required before sending data to a hosted provider."
            )
        eng = get_engine()
        provider = eng.multimodal
        st = provider.status()
        if not st.available or not isinstance(provider, MetaModelApiNarrator):
            raise api_error(
                409, "PROVIDER_UNAVAILABLE", st.reason_unavailable or "No narrative provider is configured."
            )
        payload, ev_summary = _narrative_inputs(study_id)
        try:
            narrative = provider.narrate(minimize(payload, ev_summary))
        except ProviderError as exc:
            raise api_error(502, exc.code, str(exc)) from None
        out = {
            **asdict(narrative),
            "created_at": datetime.now(UTC).isoformat(),
            "note": "Narrative text only; scores, masks and status are unchanged.",
        }
        data = json.dumps(out).encode()
        with sf() as s:
            key = store.put(study_id, data, "application/json")
            s.add(
                Artifact(
                    study_id=study_id,
                    kind="narrative",
                    storage_key=key,
                    media_type="application/json",
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            )
            s.commit()
        return out

    app.include_router(v1)
    return app


def _num(v: object) -> float:
    return float(v) if isinstance(v, int | float) else 0.0


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


def get_app() -> FastAPI:  # uvicorn factory: `uvicorn ovascope_api.main:get_app --factory`
    return create_app()
