"""HTTP surface of the assistant: routes, dependencies, error mapping, health."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from .assistant import Assistant
from .cache import TTLCache
from .config import Settings, get_settings
from .limiter import RateLimitDependency, TokenBucketLimiter
from .llm import GeminiClient, LLMBlocked, LLMError, LLMRateLimited, LLMTimeout, LLMUnavailable
from .observability import request_id_var
from .prompts import PROMPT_VERSION
from .schemas import (
    AnswerResponse,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    LoanAskRequest,
    LoanSummarizeRequest,
    SummaryResponse,
    TransactionsAskRequest,
    TransactionsSummarizeRequest,
    TtsRequest,
)
from .tts import GeminiTTS

logger = logging.getLogger(__name__)
router = APIRouter()
_state: dict[str, Any] = {}

ERROR_RESPONSES = {
    422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 502: {"model": ErrorResponse},
    503: {"model": ErrorResponse}, 504: {"model": ErrorResponse},
}


def _build_assistant(settings: Settings) -> Assistant:
    llm = GeminiClient(settings)
    try:
        tts = GeminiTTS(settings)
    except LLMUnavailable as exc:
        logger.warning("tts_unavailable", extra={"event_fields": {"reason": str(exc)[:200]}})
        tts = None
    return Assistant(
        llm,
        TTLCache(maxsize=settings.cache_size, ttl_s=settings.cache_ttl_s),
        tts=tts,
        tts_cache=TTLCache(maxsize=64, ttl_s=settings.cache_ttl_s),
    )


def get_assistant() -> Assistant:
    """Lazy singleton; a failed build is not cached so a fixed configuration can recover."""
    if "assistant" not in _state:
        _state["assistant"] = _build_assistant(get_settings())
    return _state["assistant"]


async def rate_limit_dependency(request: Request) -> None:
    if "limiter" not in _state:
        _state["limiter"] = RateLimitDependency(TokenBucketLimiter(get_settings().rate_limit_per_minute))
    await _state["limiter"](request)


AI_ROUTE = dict(dependencies=[Depends(rate_limit_dependency)], responses=ERROR_RESPONSES)


@router.post("/transactions/ask", response_model=AnswerResponse, **AI_ROUTE)
async def transactions_ask(payload: TransactionsAskRequest, assistant: Assistant = Depends(get_assistant)) -> AnswerResponse:
    return await assistant.ask_transactions(payload, request_id=request_id_var.get(""))


@router.post("/transactions/summarize", response_model=SummaryResponse, **AI_ROUTE)
async def transactions_summarize(payload: TransactionsSummarizeRequest, assistant: Assistant = Depends(get_assistant)) -> SummaryResponse:
    return await assistant.summarize_transactions(payload, request_id=request_id_var.get(""))


@router.post("/loans/ask", response_model=AnswerResponse, **AI_ROUTE)
async def loans_ask(payload: LoanAskRequest, assistant: Assistant = Depends(get_assistant)) -> AnswerResponse:
    return await assistant.ask_loan(payload, request_id=request_id_var.get(""))


@router.post("/loans/summarize", response_model=SummaryResponse, **AI_ROUTE)
async def loans_summarize(payload: LoanSummarizeRequest, assistant: Assistant = Depends(get_assistant)) -> SummaryResponse:
    return await assistant.summarize_loan(payload, request_id=request_id_var.get(""))


@router.post("/explain", response_model=ExplainResponse, **AI_ROUTE)
async def explain(payload: ExplainRequest, assistant: Assistant = Depends(get_assistant)) -> ExplainResponse:
    return await assistant.explain(payload, request_id=request_id_var.get(""))


@router.post(
    "/tts",
    dependencies=[Depends(rate_limit_dependency)],
    responses={**ERROR_RESPONSES, 200: {"content": {"audio/wav": {}}, "description": "WAV audio"}},
    response_class=Response,
)
async def tts(payload: TtsRequest, assistant: Assistant = Depends(get_assistant)) -> Response:
    wav, model, cached = await assistant.synthesize(payload, request_id=request_id_var.get(""))
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"X-Cache": "hit" if cached else "miss", "X-TTS-Model": model, "Cache-Control": "no-store"},
    )


def _assistant_or_error(request: Request) -> tuple[Assistant | None, str | None]:
    provider = request.app.dependency_overrides.get(get_assistant, get_assistant)
    try:
        return provider(), None
    except LLMUnavailable as exc:
        return None, str(exc)


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    settings = get_settings()
    assistant, error = _assistant_or_error(request)
    if assistant is None:
        llm_info: dict[str, Any] = {"configured": False, "error": error}
        tts_info: dict[str, Any] = {"configured": False, "error": error}
        cache_stats = {"size": 0, "hits": 0, "misses": 0}
    else:
        llm_info = {
            "configured": True,
            "active_model": assistant.llm.active_model,
            "models": list(getattr(assistant.llm, "models", [])),
            "dead_models": sorted(getattr(assistant.llm, "dead_models", [])),
            "location": settings.location,
            "thinking_level": settings.thinking_level,
        }
        if assistant.tts is None:
            tts_info = {"configured": False}
        else:
            tts_info = {
                "configured": True,
                "active_model": assistant.tts.active_model,
                "models": list(getattr(assistant.tts, "models", [])),
                "voice": settings.tts_voice,
                "cache": assistant.tts_cache.stats(),
            }
        cache_stats = assistant.cache.stats()
    return {
        "status": "ok",
        "request_id": request_id_var.get(""),
        "prompt_version": PROMPT_VERSION,
        "llm": llm_info,
        "tts": tts_info,
        "cache": cache_stats,
        "rate_limit": {"per_minute": settings.rate_limit_per_minute},
    }


# What the page is told. The detail (file paths, project and model resource names, SDK text) stays in the logs.
CLIENT_MESSAGES = {
    "rate_limited": "The assistant is busy right now. Please try again in a few seconds.",
    "llm_timeout": "The assistant took too long to respond. Please try again.",
    "llm_unavailable": "The assistant is not available right now. Please try again later.",
    "llm_blocked": "The assistant could not produce an answer for that request.",
    "llm_error": "The assistant could not complete that request. Please try again.",
}


def _error(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code, "message": message}, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    mapping = [
        (LLMRateLimited, 429, "rate_limited"),
        (LLMTimeout, 504, "llm_timeout"),
        (LLMUnavailable, 503, "llm_unavailable"),
        (LLMBlocked, 502, "llm_blocked"),
        (LLMError, 502, "llm_error"),
    ]
    for exc_type, status, code in mapping:
        def make_handler(status: int = status, code: str = code):
            async def handler(request: Request, exc: Exception) -> JSONResponse:
                logger.warning("ai_error", extra={"event_fields": {"error": code, "status": status, "detail": str(exc)[:300]}})
                return _error(status, code, CLIENT_MESSAGES[code])
            return handler
        app.add_exception_handler(exc_type, make_handler())

    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = ".".join(str(part) for part in first.get("loc", []) if part != "body")
        msg = str(first.get("msg", "invalid request"))
        return _error(422, "invalid_request", f"{loc}: {msg}" if loc else msg)

    async def http_handler(request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            return JSONResponse(status_code=exc.status_code, content=exc.detail, headers=exc.headers)
        return _error(exc.status_code, "http_error", str(exc.detail), headers=exc.headers)

    app.add_exception_handler(RequestValidationError, validation_handler)
    app.add_exception_handler(HTTPException, http_handler)
