"""Job lifecycle shared by the inline worker thread and services/worker.

Claiming uses a compare-and-set UPDATE (status queued -> running), which is
safe on both SQLite and PostgreSQL without Redis.
"""

from __future__ import annotations

import hashlib
import logging
import os
import socket
import threading
import time
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from ovascope_api.db import Artifact, Job, Result, Study, WorkerHeartbeat, utcnow
from ovascope_ml.inference.pipeline import AnalysisEngine, Cancelled
from ovascope_ml.inference.result_schema import ArtifactRef, ResultStatus
from ovascope_ml.providers.base import ArtifactStore

log = logging.getLogger("ovascope.jobs")

RETRYABLE_CODES = {"PROVIDER_TIMEOUT"}


def engine_fingerprint(engine: AnalysisEngine) -> str:
    if engine.mode() != "RESEARCH_INFERENCE" or engine.bundle is None:
        return f"no_model:{engine.mode()}"
    return f"bundle:{engine.bundle.checkpoint_sha256[:32]}"


def claim_next(sf: sessionmaker, worker_id: str) -> str | None:
    with sf() as s:
        candidates = s.scalars(select(Job.id).where(Job.status == "queued").order_by(Job.created_at).limit(5)).all()
        for job_id in candidates:
            res = s.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == "queued")
                .values(
                    status="running",
                    worker_id=worker_id,
                    started_at=utcnow(),
                    stage="claimed",
                    attempts=Job.attempts + 1,
                )
            )
            s.commit()
            if res.rowcount == 1:
                return job_id
    return None


def _set(sf: sessionmaker, job_id: str, **values: object) -> None:
    with sf() as s:
        s.execute(update(Job).where(Job.id == job_id).values(**values))
        s.commit()


def process_job(sf: sessionmaker, store: ArtifactStore, engine: AnalysisEngine, job_id: str) -> None:
    with sf() as s:
        job = s.get(Job, job_id)
        if job is None:
            return
        study = s.get(Study, job.study_id)
        if study is None or job.cancel_requested:
            job.status, job.finished_at, job.stage = "cancelled", utcnow(), None
            s.commit()
            return
        image_key, study_id, attempts, max_attempts = study.image_key, study.id, job.attempts, job.max_attempts

    def cancelled() -> bool:
        with sf() as s2:
            j = s2.get(Job, job_id)
            return j is None or j.cancel_requested

    def on_stage(name: str) -> None:
        _set(sf, job_id, stage=name)

    try:
        data = store.get(image_key)
        out = engine.analyze(data, study_id, cancel_check=cancelled, on_stage=on_stage)
    except Cancelled:
        _set(sf, job_id, status="cancelled", stage=None, finished_at=utcnow())
        log.info("job cancelled", extra={"job_id": job_id})
        return
    except Exception as exc:
        log.exception("job crashed", extra={"job_id": job_id})
        retry = attempts < max_attempts
        _set(
            sf,
            job_id,
            status="queued" if retry else "failed",
            stage=None,
            error_code="WORKER_ERROR",
            error_message=f"{type(exc).__name__}",
            finished_at=None if retry else utcnow(),
        )
        return

    r = out.result
    if r.status == ResultStatus.failed and r.error and r.error.code in RETRYABLE_CODES and attempts < max_attempts:
        _set(sf, job_id, status="queued", stage=None, error_code=r.error.code, error_message=r.error.message)
        log.info("job requeued", extra={"job_id": job_id, "error_code": r.error.code, "attempt": attempts})
        return

    with sf() as s:
        if s.get(Study, study_id) is None:  # deleted while running
            store.delete_namespace(study_id)
            return
        for art in out.artifacts:
            key = store.put(study_id, art.png, "image/png")
            row = Artifact(
                study_id=study_id,
                kind=art.kind,
                storage_key=key,
                media_type="image/png",
                sha256=hashlib.sha256(art.png).hexdigest(),
            )
            s.add(row)
            s.flush()
            ref = ArtifactRef(
                artifact_id=row.id,
                kind=art.kind,
                width=art.width,
                height=art.height,
                source=art.source,
                description=art.description,
                limitations=art.limitations,
            )
            if art.kind == "segmentation_mask":
                r.segmentation_artifact_or_null = ref
            elif art.kind == "attribution_map":
                r.attribution_artifact_or_null = ref
            elif art.kind == "localizer_prior":
                r.localizer_artifact_or_null = ref
        s.add(Result(study_id=study_id, job_id=job_id, status=r.status.value, payload=r.model_dump(mode="json")))
        job = s.get(Job, job_id)
        assert job is not None
        job.status = "failed" if r.status == ResultStatus.failed else "succeeded"
        job.error_code = r.error.code if r.error else None
        job.error_message = r.error.message if r.error else None
        job.stage, job.finished_at = None, utcnow()
        s.commit()
    log.info("job finished", extra={"job_id": job_id, "status": r.status.value})


def purge_expired(sf: sessionmaker, store: ArtifactStore) -> int:
    now = datetime.now(UTC)
    with sf() as s:
        expired = s.scalars(select(Study).where(Study.retention_expires_at < now)).all()
        for st in expired:
            store.delete_namespace(st.id)
            s.delete(st)
        s.commit()
        return len(expired)


def heartbeat(sf: sessionmaker, worker_id: str, mode: str, inline: bool) -> None:
    with sf() as s:
        hb = s.get(WorkerHeartbeat, worker_id)
        if hb is None:
            s.add(WorkerHeartbeat(worker_id=worker_id, mode=mode, inline=inline, last_seen=utcnow()))
        else:
            hb.last_seen, hb.mode, hb.inline = utcnow(), mode, inline
        s.commit()


def latest_heartbeat(s: Session) -> WorkerHeartbeat | None:
    return s.scalars(select(WorkerHeartbeat).order_by(WorkerHeartbeat.last_seen.desc()).limit(1)).first()


def requeue_stale(sf: sessionmaker, older_than: timedelta = timedelta(minutes=15)) -> int:
    """Jobs left 'running' by a crashed worker go back to the queue (bounded by max_attempts)."""
    cutoff = datetime.now(UTC) - older_than
    with sf() as s:
        stale = s.scalars(select(Job).where(Job.status == "running", Job.updated_at < cutoff)).all()
        for j in stale:
            j.status = "queued" if j.attempts < j.max_attempts else "failed"
            j.stage = None
            if j.status == "failed":
                j.error_code, j.finished_at = "WORKER_LOST", utcnow()
        s.commit()
        return len(stale)


class WorkerLoop:
    def __init__(
        self, sf: sessionmaker, store: ArtifactStore, engine_getter, inline: bool, poll_s: float = 0.5
    ) -> None:
        self.sf, self.store, self.engine_getter, self.inline, self.poll_s = sf, store, engine_getter, inline, poll_s
        self.worker_id = f"{'inline' if inline else 'worker'}-{socket.gethostname()[:20]}-{os.getpid()}"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_forever(self) -> None:
        last_maint = 0.0
        while not self._stop.is_set():
            engine = self.engine_getter()
            heartbeat(self.sf, self.worker_id, engine.mode(), self.inline)
            if time.monotonic() - last_maint > 60:
                purge_expired(self.sf, self.store)
                requeue_stale(self.sf)
                last_maint = time.monotonic()
            job_id = claim_next(self.sf, self.worker_id)
            if job_id is None:
                self._stop.wait(self.poll_s)
                continue
            process_job(self.sf, self.store, engine, job_id)

    def start_thread(self) -> None:
        self._thread = threading.Thread(target=self.run_forever, name="ovascope-inline-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
