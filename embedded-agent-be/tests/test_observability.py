import json
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.observability import JsonFormatter, RequestIdMiddleware, log_event, request_id_var


def make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/x")
    async def x():
        return {"rid": request_id_var.get()}

    return app


def test_request_id_header_round_trip():
    client = TestClient(make_app())
    response = client.get("/x", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
    assert response.json()["rid"] == "abc-123"
    generated = client.get("/x")
    rid = generated.headers["X-Request-ID"]
    assert len(rid) >= 8
    assert generated.json()["rid"] == rid


def test_request_id_header_is_replaced_when_malformed():
    client = TestClient(make_app())
    response = client.get("/x", headers={"X-Request-ID": "x" * 200})
    assert response.headers["X-Request-ID"] != "x" * 200
    response = client.get("/x", headers={"X-Request-ID": "bad id!"})
    assert response.headers["X-Request-ID"] != "bad id!"


def test_json_formatter_includes_request_id_and_fields():
    record = logging.LogRecord("agent.test", logging.INFO, __file__, 1, "ai_request", None, None)
    record.event_fields = {"page": "transactions", "latency_ms": 12}
    token = request_id_var.set("rid-1")
    try:
        out = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert out["request_id"] == "rid-1"
    assert out["event"] == "ai_request"
    assert out["level"] == "INFO"
    assert out["page"] == "transactions"
    assert out["latency_ms"] == 12
    assert "ts" in out


def test_json_formatter_without_request_id_or_fields():
    record = logging.LogRecord("agent.test", logging.WARNING, __file__, 1, "plain message", None, None)
    out = json.loads(JsonFormatter().format(record))
    assert out["request_id"] == ""
    assert out["event"] == "plain message"
    assert out["level"] == "WARNING"


def test_log_event_emits_record_with_fields(caplog):
    with caplog.at_level(logging.INFO, logger="agent"):
        log_event("ai_request", page="loan", outcome="ok")
    record = caplog.records[-1]
    assert record.getMessage() == "ai_request"
    assert record.event_fields == {"page": "loan", "outcome": "ok"}
