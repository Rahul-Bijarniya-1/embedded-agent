"""Gemini text client: async, ordered model fallback, retry policy, thinking step-down.

Also the FakeLLM used by every test that does not talk to Google.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .config import Settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """A model call failed for a reason the caller cannot fix by retrying."""


class LLMUnavailable(LLMError):
    """No usable model / client: configuration, credentials, or persistent 5xx."""


class LLMRateLimited(LLMError):
    """429 after retries."""


class LLMTimeout(LLMError):
    """The call exceeded the configured timeout."""


class LLMBlocked(LLMError):
    """The model returned no usable content (safety filter or empty candidate)."""


class _ModelNotFound(LLMError):
    """Internal: the model id does not exist at this location; try the next one."""


RETRYABLE_CODES = {429, 500, 502, 503, 504}
BACKOFF_S = (0.5, 1.5)
THINKING_STEP_DOWN: dict[str, str | None] = {"minimal": "low", "low": None}


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLM(Protocol):
    active_model: str

    async def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_output_tokens: int,
        response_schema: type | None = None,
    ) -> LLMResult: ...


def describe_error(exc: BaseException) -> tuple[int | None, str]:
    """(http status code or None, message) for any exception the SDK may raise."""
    raw_code = getattr(exc, "code", None)
    try:
        code = int(raw_code) if raw_code is not None else None
    except (TypeError, ValueError):
        code = None
    message = getattr(exc, "message", None) or str(exc)
    return code, str(message)


def is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return True
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text


class GeminiBase:
    """Shared machinery for the text and TTS clients."""

    def __init__(self, settings: Settings, *, models: list[str], location: str, timeout_s: float):
        self.settings = settings
        self.models = list(models)
        self.location = location
        self.timeout_s = timeout_s
        self.dead_models: set[str] = set()
        if not settings.project:
            raise LLMUnavailable("GOOGLE_CLOUD_PROJECT is not set; see .env.example")
        if not self.models:
            raise LLMUnavailable("No Gemini model id configured")
        try:
            self._client = self._make_client()
        except Exception as exc:  # noqa: BLE001 - any failure here means "not configured"
            raise LLMUnavailable(f"Gemini client could not be created: {exc}") from exc

    def _make_client(self) -> Any:
        from google import genai
        from google.genai import types

        return genai.Client(
            vertexai=True,
            project=self.settings.project,
            location=self.location,
            http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
        )

    @property
    def active_model(self) -> str:
        for model in self.models:
            if model not in self.dead_models:
                return model
        return "none"

    async def _sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    def _step_down_thinking(self, model: str) -> bool:
        return False

    async def _with_fallback(self, attempt: Callable[[str], Awaitable[Any]]) -> Any:
        last: Exception | None = None
        for model in self.models:
            if model in self.dead_models:
                continue
            try:
                return await self._with_retries(model, attempt)
            except _ModelNotFound as exc:
                self.dead_models.add(model)
                last = exc
                logger.warning("model_fallback", extra={"event_fields": {"dead_model": model, "reason": str(exc)[:200]}})
        raise LLMUnavailable(
            f"No configured Gemini model is available ({', '.join(self.models)})"
            + (f": {last}" if last else "")
        )

    async def _with_retries(self, model: str, attempt: Callable[[str], Awaitable[Any]]) -> Any:
        tries = 0
        while True:
            try:
                return await attempt(model)
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                code, message = describe_error(exc)
                if code is None and is_timeout(exc):
                    raise LLMTimeout(f"Gemini call timed out after {self.timeout_s:.0f}s") from exc
                if code == 404 or "not found" in message.lower():
                    raise _ModelNotFound(message) from exc
                if code == 400 and "thinking" in message.lower() and self._step_down_thinking(model):
                    continue
                if code in RETRYABLE_CODES and tries < len(BACKOFF_S):
                    await self._sleep(BACKOFF_S[tries])
                    tries += 1
                    continue
                if code == 429:
                    raise LLMRateLimited("Gemini rate limit exceeded; try again shortly") from exc
                if code in RETRYABLE_CODES:
                    raise LLMUnavailable(f"Gemini unavailable ({code}): {message}") from exc
                raise LLMError(f"Gemini request failed ({code}): {message}") from exc


class GeminiClient(GeminiBase):
    """Text generation through Vertex AI."""

    def __init__(self, settings: Settings):
        super().__init__(
            settings, models=settings.models, location=settings.location, timeout_s=settings.timeout_s
        )
        self._thinking: dict[str, str | None] = {}

    def thinking_level_for(self, model: str) -> str | None:
        return self._thinking.get(model, self.settings.thinking_level)

    def _step_down_thinking(self, model: str) -> bool:
        current = self.thinking_level_for(model)
        if current is None:
            return False
        self._thinking[model] = THINKING_STEP_DOWN.get(current.lower())
        logger.warning(
            "thinking_step_down",
            extra={"event_fields": {"model": model, "from": current, "to": self._thinking[model]}},
        )
        return True

    def _config(self, model: str, temperature: float, max_output_tokens: int, response_schema: type | None) -> dict[str, Any]:
        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "thinking_level": self.thinking_level_for(model),
        }
        if response_schema is not None:
            config["response_schema"] = response_schema
            config["response_mime_type"] = "application/json"
        return config

    async def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_output_tokens: int,
        response_schema: type | None = None,
    ) -> LLMResult:
        async def attempt(model: str) -> LLMResult:
            config = self._config(model, temperature, max_output_tokens, response_schema)
            response = await self._call_once(model, system, user, config)
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise LLMBlocked("Gemini returned no text (safety filter or empty candidate)")
            usage = getattr(response, "usage_metadata", None)
            return LLMResult(
                text=text,
                model=model,
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
            )

        return await self._with_fallback(attempt)

    async def _call_once(self, model: str, system: str, user: str, config: dict[str, Any]) -> Any:
        from google.genai import types

        kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": config["temperature"],
            "max_output_tokens": config["max_output_tokens"],
        }
        if config.get("thinking_level"):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=config["thinking_level"])
        if config.get("response_schema") is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = config["response_schema"]
        return await self._client.aio.models.generate_content(
            model=model, contents=user, config=types.GenerateContentConfig(**kwargs)
        )


class FakeLLM:
    """Scripted stand-in for tests: records calls, replies in sequence, or raises."""

    def __init__(self, reply: str = "FAKE", error: Exception | None = None, replies: list[str] | None = None):
        self.reply = reply
        self.error = error
        self.replies = list(replies) if replies else None
        self.calls: list[dict[str, Any]] = []
        self.active_model = "fake-model"
        self.dead_models: set[str] = set()

    async def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_output_tokens: int,
        response_schema: type | None = None,
    ) -> LLMResult:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "response_schema": response_schema,
            }
        )
        if self.error:
            raise self.error
        text = self.reply
        if self.replies:
            text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return LLMResult(text=text, model=self.active_model, input_tokens=10, output_tokens=5)
