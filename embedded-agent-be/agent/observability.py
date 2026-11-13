"""Request ids and structured JSON logs. No external services; stdout is the sink."""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextvars import ContextVar
from typing import Any

from starlette.datastructures import Headers, MutableHeaders

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_logger = logging.getLogger("agent")


class RequestIdMiddleware:
    """Pure ASGI middleware: honour a well-formed X-Request-ID or mint one; echo it back."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = Headers(scope=scope).get("x-request-id", "")
        request_id = incoming if _VALID_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)

        async def send_with_id(message: dict) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["x-request-id"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(""),
            "event": record.getMessage(),
        }
        fields = getattr(record, "event_fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_agent_json", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler._agent_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "google", "google_genai", "google_genai.models", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(event: str, **fields: Any) -> None:
    _logger.info(event, extra={"event_fields": fields})
