import struct

import httpx
import pytest
from google.genai import errors

from agent.config import Settings
from agent.llm import LLMBlocked, LLMTimeout, LLMUnavailable
from agent.tts import FakeTTS, GeminiTTS, TTSResult, ensure_wav


def make_settings(**overrides) -> Settings:
    values = dict(project="proj", gemini_tts_model="t1,t2", tts_voice="Kore", credentials_path=None)
    values.update(overrides)
    return Settings(_env_file=None, **values)


def api_error(code: int, message: str) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": message, "status": "X"}})


@pytest.fixture
def tts(monkeypatch):
    monkeypatch.setattr(GeminiTTS, "_make_client", lambda self: object())
    t = GeminiTTS(make_settings())
    t.sleeps = []

    async def fake_sleep(seconds):
        t.sleeps.append(seconds)

    monkeypatch.setattr(t, "_sleep", fake_sleep)
    return t


def script(tts, outcomes):
    calls = []
    queue = list(outcomes)

    async def fake_call(model, text, voice):
        calls.append({"model": model, "text": text, "voice": voice})
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    tts._call_once = fake_call
    return calls


def test_ensure_wav_passthrough_riff():
    data = b"RIFF" + b"\x00" * 40
    assert ensure_wav(data, "audio/wav") == data
    assert ensure_wav(data, "") == data


def test_ensure_wav_wraps_pcm():
    pcm = b"\x01\x02" * 100
    wav = ensure_wav(pcm, "audio/L16;codec=pcm;rate=24000")
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert struct.unpack("<H", wav[22:24])[0] == 1
    assert struct.unpack("<I", wav[24:28])[0] == 24000
    assert struct.unpack("<H", wav[34:36])[0] == 16
    assert struct.unpack("<I", wav[40:44])[0] == len(pcm)
    assert len(wav) == 44 + len(pcm)


def test_ensure_wav_reads_rate_from_mime():
    wav = ensure_wav(b"\x00\x00" * 10, "audio/pcm;rate=16000")
    assert struct.unpack("<I", wav[24:28])[0] == 16000


async def test_fake_tts_returns_wav_and_records_calls():
    fake = FakeTTS()
    result = await fake.synthesize("hello")
    assert isinstance(result, TTSResult)
    assert result.wav[:4] == b"RIFF"
    assert result.model == "fake-tts"
    assert fake.active_model == "fake-tts"
    assert fake.calls == ["hello"]
    with pytest.raises(LLMTimeout):
        await FakeTTS(error=LLMTimeout("slow")).synthesize("x")


async def test_tts_returns_wav_from_first_model(tts):
    calls = script(tts, [(b"RIFF1234", "audio/wav")])
    result = await tts.synthesize("Hello there")
    assert result == TTSResult(wav=b"RIFF1234", model="t1")
    assert calls == [{"model": "t1", "text": "Hello there", "voice": "Kore"}]


async def test_tts_wraps_pcm_output(tts):
    script(tts, [(b"\x00\x01" * 10, "audio/L16;codec=pcm;rate=24000")])
    result = await tts.synthesize("x")
    assert result.wav[:4] == b"RIFF"
    assert len(result.wav) == 44 + 20


async def test_tts_falls_back_on_not_found(tts):
    calls = script(tts, [api_error(404, "Publisher model `t1` was not found"), (b"RIFFok", "audio/wav")])
    result = await tts.synthesize("x")
    assert result.model == "t2"
    assert tts.active_model == "t2"
    assert tts.dead_models == {"t1"}
    assert [c["model"] for c in calls] == ["t1", "t2"]


async def test_tts_all_models_dead_is_unavailable(tts):
    script(tts, [api_error(404, "nope"), api_error(404, "nope")])
    with pytest.raises(LLMUnavailable):
        await tts.synthesize("x")


async def test_tts_timeout_maps_to_llm_timeout(tts):
    script(tts, [httpx.ReadTimeout("slow")])
    with pytest.raises(LLMTimeout):
        await tts.synthesize("x")


async def test_tts_retries_503_then_succeeds(tts):
    calls = script(tts, [api_error(503, "busy"), (b"RIFFok", "audio/wav")])
    assert (await tts.synthesize("x")).wav == b"RIFFok"
    assert len(calls) == 2
    assert tts.sleeps == [0.5]


async def test_tts_empty_audio_is_blocked(tts):
    script(tts, [(b"", "audio/wav")])
    with pytest.raises(LLMBlocked):
        await tts.synthesize("x")
