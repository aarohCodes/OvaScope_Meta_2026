"""Structured JSON logging with redaction.

Never logged: request/response bodies, image bytes, upload file names, auth
headers, tokens. Log records may carry only allow-listed structured fields.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from datetime import UTC, datetime

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

ALLOWED_EXTRA = {
    "study_id",
    "job_id",
    "status",
    "path",
    "method",
    "status_code",
    "duration_ms",
    "stage",
    "error_code",
    "worker_id",
    "mode",
    "attempt",
}
_BEARER = re.compile(r"(?i)bearer\s+[a-z0-9._\-]+")
_SECRETISH = re.compile(r"(?i)(token|secret|password|authorization)=\S+")


def redact(text: str) -> str:
    text = _BEARER.sub("Bearer [REDACTED]", text)
    return _SECRETISH.sub(r"\1=[REDACTED]", text)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage())[:2000],
            "request_id": request_id_var.get(),
        }
        for k in ALLOWED_EXTRA:
            if hasattr(record, k):
                out[k] = getattr(record, k)
        if record.exc_info:
            # exception type only: messages/tracebacks may embed user data
            out["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(out, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).handlers[:] = []
        logging.getLogger(noisy).propagate = False
