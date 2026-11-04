"""Standalone analysis worker: polls the jobs table, runs the engine, stores results.

Run with OVASCOPE_INLINE_WORKER=false on the API so exactly one process executes jobs.
"""

from __future__ import annotations

import logging
import signal
from pathlib import Path

from ovascope_api import jobs as jobs_mod
from ovascope_api.db import make_engine, make_session_factory
from ovascope_api.logging_setup import configure_logging
from ovascope_api.migrate import upgrade
from ovascope_api.settings import get_settings
from ovascope_ml.inference.pipeline import AnalysisEngine
from ovascope_ml.providers.artifact_store import LocalArtifactStore

log = logging.getLogger("ovascope.worker")


def main() -> int:
    configure_logging()
    settings = get_settings()
    settings.assert_safe()
    if settings.auto_migrate:
        upgrade(settings.database_url)
    sf = make_session_factory(make_engine(settings.database_url))
    store = LocalArtifactStore(Path(settings.artifact_dir))
    engine = AnalysisEngine.from_environment()
    log.info("worker starting", extra={"mode": engine.mode()})
    loop = jobs_mod.WorkerLoop(sf, store, lambda: engine, inline=False, poll_s=1.0)
    signal.signal(signal.SIGTERM, lambda *_: loop.stop())
    signal.signal(signal.SIGINT, lambda *_: loop.stop())
    loop.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
