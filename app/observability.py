"""Structured JSON logging that Cloud Logging parses natively.

On Cloud Run, anything printed to stdout as JSON with `severity` and `message`
becomes a structured log entry; the `logging.googleapis.com/trace` field links
logs to the request trace. These fields power the log-based metrics in
observability/create_metrics.sh.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "time": datetime.now(timezone.utc).isoformat(),
        }
        payload.update(getattr(record, "fields", {}))
        trace = payload.pop("trace_id", None)
        if trace and PROJECT:
            payload["logging.googleapis.com/trace"] = f"projects/{PROJECT}/traces/{trace}"
        return json.dumps(payload, default=str)


def get_logger(name: str = "risk-assistant") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, event, extra={"fields": {"event": event, **fields}})


def trace_id_from_header(header_value: str | None) -> str | None:
    """X-Cloud-Trace-Context looks like 'TRACE_ID/SPAN_ID;o=1'."""
    if not header_value:
        return None
    return header_value.split("/")[0] or None
