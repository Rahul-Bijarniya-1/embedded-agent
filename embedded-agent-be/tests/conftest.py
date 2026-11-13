import json
import pathlib

import pytest
from fastapi.testclient import TestClient

import server
from agent import router as router_module
from agent.assistant import Assistant
from agent.cache import TTLCache
from agent.llm import FakeLLM
from agent.tts import FakeTTS

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(reply="FAKE ANSWER")


@pytest.fixture
def fake_tts() -> FakeTTS:
    return FakeTTS()


@pytest.fixture
def assistant(fake_llm, fake_tts) -> Assistant:
    return Assistant(
        fake_llm,
        TTLCache(maxsize=64, ttl_s=600),
        tts=fake_tts,
        tts_cache=TTLCache(maxsize=8, ttl_s=600),
    )


@pytest.fixture
def client(assistant):
    app = server.app
    app.dependency_overrides[router_module.get_assistant] = lambda: assistant
    app.dependency_overrides[router_module.rate_limit_dependency] = lambda: None
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()
