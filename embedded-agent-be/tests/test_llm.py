from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from agent.config import Settings
from agent.llm import (
    FakeLLM,
    GeminiClient,
    LLMBlocked,
    LLMError,
    LLMRateLimited,
    LLMResult,
    LLMTimeout,
    LLMUnavailable,
)
from agent.prompts import AskOutput


def make_settings(**overrides) -> Settings:
    values = dict(project="proj", gemini_model="m1,m2", thinking_level="minimal", credentials_path=None)
    values.update(overrides)
    return Settings(_env_file=None, **values)


class Resp:
    def __init__(self, text, prompt_tokens=10, out_tokens=5):
        self.text = text
        self.usage_metadata = SimpleNamespace(prompt_token_count=prompt_tokens, candidates_token_count=out_tokens)


def api_error(code: int, message: str) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": message, "status": "X"}})


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(GeminiClient, "_make_client", lambda self: object())
    c = GeminiClient(make_settings())
    c.sleeps = []

    async def fake_sleep(seconds):
        c.sleeps.append(seconds)

    monkeypatch.setattr(c, "_sleep", fake_sleep)
    return c


def script(client, outcomes):
    """Replace _call_once with a scripted sequence; returns the recorded calls."""
    calls = []
    queue = list(outcomes)

    async def fake_call(model, system, user, config):
        calls.append({"model": model, "system": system, "user": user, "config": dict(config)})
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client._call_once = fake_call
    return calls


async def generate(client, **kwargs):
    params = dict(temperature=0.2, max_output_tokens=50)
    params.update(kwargs)
    return await client.generate("sys", "usr", **params)


async def test_fake_llm_records_calls():
    llm = FakeLLM(reply="hi")
    result = await llm.generate("sys", "usr", temperature=0.1, max_output_tokens=10)
    assert result.text == "hi"
    assert result.model == "fake-model"
    assert llm.active_model == "fake-model"
    assert llm.calls == [
        {"system": "sys", "user": "usr", "temperature": 0.1, "max_output_tokens": 10, "response_schema": None}
    ]


async def test_fake_llm_replies_in_sequence_and_raises_configured_error():
    llm = FakeLLM(replies=["one", "two"])
    assert (await llm.generate("s", "u", temperature=0, max_output_tokens=1)).text == "one"
    assert (await llm.generate("s", "u", temperature=0, max_output_tokens=1)).text == "two"
    assert (await llm.generate("s", "u", temperature=0, max_output_tokens=1)).text == "two"
    with pytest.raises(LLMTimeout):
        await FakeLLM(error=LLMTimeout("slow")).generate("s", "u", temperature=0, max_output_tokens=1)


def test_gemini_client_init_failure_is_llm_unavailable(monkeypatch):
    def boom(self):
        raise RuntimeError("could not load credentials")

    monkeypatch.setattr(GeminiClient, "_make_client", boom)
    with pytest.raises(LLMUnavailable):
        GeminiClient(make_settings())


async def test_generate_returns_text_usage_and_passes_config(client):
    calls = script(client, [Resp("hello")])
    result = await generate(client)
    assert result == LLMResult(text="hello", model="m1", input_tokens=10, output_tokens=5)
    assert calls[0]["model"] == "m1"
    assert calls[0]["system"] == "sys" and calls[0]["user"] == "usr"
    assert calls[0]["config"]["temperature"] == 0.2
    assert calls[0]["config"]["max_output_tokens"] == 50
    assert calls[0]["config"]["thinking_level"] == "minimal"
    assert client.active_model == "m1"


async def test_response_schema_is_passed_through_as_json_mode(client):
    calls = script(client, [Resp('{"answer": "x", "scope": "in_page"}')])
    await generate(client, response_schema=AskOutput)
    assert calls[0]["config"]["response_schema"] is AskOutput
    assert calls[0]["config"]["response_mime_type"] == "application/json"


async def test_generate_retries_503_then_succeeds(client):
    calls = script(client, [api_error(503, "unavailable"), Resp("ok")])
    assert (await generate(client)).text == "ok"
    assert len(calls) == 2
    assert client.sleeps == [0.5]


async def test_generate_429_exhausts_to_rate_limited(client):
    calls = script(client, [api_error(429, "quota")] * 3)
    with pytest.raises(LLMRateLimited):
        await generate(client)
    assert len(calls) == 3
    assert client.sleeps == [0.5, 1.5]


async def test_server_errors_exhausted_is_unavailable(client):
    script(client, [api_error(500, "boom")] * 3)
    with pytest.raises(LLMUnavailable):
        await generate(client)


async def test_generate_400_not_retried(client):
    calls = script(client, [api_error(400, "bad request")])
    with pytest.raises(LLMError) as info:
        await generate(client)
    assert type(info.value) is LLMError
    assert len(calls) == 1
    assert client.sleeps == []


async def test_generate_timeout_maps_to_llm_timeout(client):
    script(client, [httpx.ReadTimeout("slow")])
    with pytest.raises(LLMTimeout):
        await generate(client)


async def test_generate_empty_text_is_blocked(client):
    script(client, [Resp(None)])
    with pytest.raises(LLMBlocked):
        await generate(client)
    script(client, [Resp("   ")])
    with pytest.raises(LLMBlocked):
        await generate(client)


async def test_generate_falls_back_on_not_found(client):
    calls = script(client, [api_error(404, "Publisher model `m1` was not found"), Resp("from m2")])
    result = await generate(client)
    assert result.model == "m2"
    assert client.active_model == "m2"
    assert client.dead_models == {"m1"}
    assert [c["model"] for c in calls] == ["m1", "m2"]
    later = script(client, [Resp("again")])
    await generate(client)
    assert [c["model"] for c in later] == ["m2"]


async def test_all_models_dead_is_unavailable(client):
    script(client, [api_error(404, "not found"), api_error(404, "not found")])
    with pytest.raises(LLMUnavailable):
        await generate(client)
    assert client.dead_models == {"m1", "m2"}
    with pytest.raises(LLMUnavailable):
        await generate(client)


async def test_thinking_rejected_steps_down_and_is_remembered(client):
    calls = script(
        client,
        [
            api_error(400, "Thinking level is unsupported: THINKING_LEVEL_MINIMAL"),
            api_error(400, "Thinking level is unsupported: THINKING_LEVEL_LOW"),
            Resp("ok"),
        ],
    )
    assert (await generate(client)).text == "ok"
    assert [c["config"].get("thinking_level") for c in calls] == ["minimal", "low", None]
    assert client.sleeps == []
    later = script(client, [Resp("ok2")])
    await generate(client)
    assert later[0]["config"].get("thinking_level") is None


async def test_no_thinking_level_configured_sends_none(monkeypatch):
    monkeypatch.setattr(GeminiClient, "_make_client", lambda self: object())
    c = GeminiClient(make_settings(thinking_level=""))
    calls = script(c, [Resp("ok")])
    await generate(c)
    assert calls[0]["config"].get("thinking_level") is None
