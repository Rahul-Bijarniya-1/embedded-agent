"""Gemini text-to-speech through Vertex AI, returning WAV bytes; plus a fake for tests."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings
from .llm import GeminiBase, LLMBlocked

FAKE_WAV = b"RIFF" + b"\x00" * 40


@dataclass(frozen=True)
class TTSResult:
    wav: bytes
    model: str


class TTS(Protocol):
    active_model: str

    async def synthesize(self, text: str) -> TTSResult: ...


def ensure_wav(data: bytes, mime_type: str) -> bytes:
    """Return WAV bytes; raw PCM (audio/L16 or audio/pcm, 16-bit mono) gets a 44-byte header."""
    mime = (mime_type or "").lower()
    if data[:4] == b"RIFF" or mime.startswith(("audio/wav", "audio/x-wav")):
        return data
    rate = 24000
    for piece in mime.split(";"):
        piece = piece.strip()
        if piece.startswith("rate="):
            try:
                rate = int(piece[5:])
            except ValueError:
                pass
    channels, bits = 1, 16
    byte_rate = rate * channels * bits // 8
    block_align = channels * bits // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 1, channels, rate, byte_rate, block_align, bits,
        b"data", len(data),
    )
    return header + data


class GeminiTTS(GeminiBase):
    def __init__(self, settings: Settings):
        super().__init__(
            settings,
            models=settings.tts_models,
            location=settings.tts_location,
            timeout_s=settings.tts_timeout_s,
        )

    async def synthesize(self, text: str) -> TTSResult:
        async def attempt(model: str) -> TTSResult:
            data, mime = await self._call_once(model, text, self.settings.tts_voice)
            if not data:
                raise LLMBlocked("Gemini TTS returned no audio")
            return TTSResult(wav=ensure_wav(data, mime), model=model)

        return await self._with_fallback(attempt)

    async def _call_once(self, model: str, text: str, voice: str) -> tuple[bytes, str]:
        from google.genai import types

        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))
            ),
        )
        response = await self._client.aio.models.generate_content(model=model, contents=text, config=config)
        for candidate in response.candidates or []:
            parts = (candidate.content.parts if candidate.content else None) or []
            for part in parts:
                blob = getattr(part, "inline_data", None)
                if blob is not None and blob.data:
                    return blob.data, blob.mime_type or ""
        return b"", ""


class FakeTTS:
    def __init__(self, wav: bytes = FAKE_WAV, error: Exception | None = None):
        self.wav = wav
        self.error = error
        self.calls: list[str] = []
        self.active_model = "fake-tts"
        self.dead_models: set[str] = set()

    async def synthesize(self, text: str) -> TTSResult:
        self.calls.append(text)
        if self.error:
            raise self.error
        return TTSResult(wav=self.wav, model=self.active_model)
