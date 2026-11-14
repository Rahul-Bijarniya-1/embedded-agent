"""Orchestration: page payload → context → prompt → model → parse/clean/screen → response."""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from datetime import date

from pydantic import ValidationError

from .cache import TTLCache, cache_key
from .context import PageContext, build_loan_context, build_section_context, build_transactions_context
from .llm import LLM, LLMError, LLMUnavailable
from .observability import log_event
from .prompts import (
    GENERATION_PARAMS,
    GROUNDING_RULE,
    INJECTION_RULE,
    PROMPT_VERSION,
    Action,
    AskOutput,
    build_prompt,
)
from .schemas import (
    AnswerResponse,
    ExplainRequest,
    ExplainResponse,
    LoanAskRequest,
    LoanSummarizeRequest,
    Meta,
    Scope,
    SummaryResponse,
    TransactionsAskRequest,
    TransactionsSummarizeRequest,
    TtsRequest,
)
from .tts import TTS

SCREENED_MESSAGE = (
    "I couldn't produce a safe answer for that request. Please ask about the data shown on this page."
)
LEAK_MARKERS = [
    "<page_data>",
    "<metadata>",
    "<user_question>",
    "<section_text>",
    "page data.facts",
    "system prompt",
    "app pages (names only",
    GROUNDING_RULE[:40],
    INJECTION_RULE[:40],
]

_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*[ \t]*\n?(.*?)\n?```$", re.DOTALL)
_LABEL = re.compile(r"^(?:\*\*)?(answer|summary|explanation)(?:\*\*)?\s*:\s*(?:\*\*)?\s*", re.IGNORECASE)
_BLANKS = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    out = (text or "").strip()
    fenced = _FENCE.match(out)
    if fenced:
        out = fenced.group(1).strip()
    out = _LABEL.sub("", out, count=1)
    out = _BLANKS.sub("\n\n", out)
    return out.strip()


def screen_output(text: str) -> tuple[str, bool]:
    """Replace an answer that quotes the prompt scaffolding; returns (text, screened)."""
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in LEAK_MARKERS):
        return SCREENED_MESSAGE, True
    return text, False


def parse_ask(raw: str) -> tuple[str, Scope]:
    body = clean_text(raw)
    try:
        parsed = AskOutput.model_validate(json.loads(body))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return body, "unknown"
    return clean_text(parsed.answer), parsed.scope


class Assistant:
    def __init__(
        self,
        llm: LLM,
        cache: TTLCache,
        tts: TTS | None = None,
        tts_cache: TTLCache | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        today: Callable[[], date] = date.today,
    ):
        self.llm = llm
        self.cache = cache
        self.tts = tts
        self.tts_cache = tts_cache if tts_cache is not None else TTLCache(maxsize=64, ttl_s=cache.ttl_s)
        self._clock = clock
        self._today = today

    # --- public actions ----------------------------------------------------------------

    async def ask_transactions(self, request: TransactionsAskRequest, request_id: str = "") -> AnswerResponse:
        context = build_transactions_context(request.transactions, locale=request.locale, today=self._today())
        return await self._ask(context, request.question, request_id)

    async def summarize_transactions(self, request: TransactionsSummarizeRequest, request_id: str = "") -> SummaryResponse:
        context = build_transactions_context(request.transactions, locale=request.locale, today=self._today())
        return await self._summarize(context, request_id)

    async def ask_loan(self, request: LoanAskRequest, request_id: str = "") -> AnswerResponse:
        context = build_loan_context(request.loan_data, locale=request.locale, today=self._today())
        return await self._ask(context, request.question, request_id)

    async def summarize_loan(self, request: LoanSummarizeRequest, request_id: str = "") -> SummaryResponse:
        context = build_loan_context(request.loan_data, locale=request.locale, today=self._today())
        return await self._summarize(context, request_id)

    async def explain(self, request: ExplainRequest, request_id: str = "") -> ExplainResponse:
        context = build_section_context(
            request.page, request.section, request.section_text, request.data,
            locale=request.locale, today=self._today(),
        )
        text, _, meta = await self._generate(context, "explain", None, request_id)
        return ExplainResponse(explanation=text, meta=meta)

    async def synthesize(self, request: TtsRequest, request_id: str = "") -> tuple[bytes, str, bool]:
        """Return (wav bytes, model id, served_from_cache) for the given text."""
        if self.tts is None:
            raise LLMUnavailable("Text-to-speech is not configured")
        started = self._clock()
        voice = getattr(self.tts, "voice", "")
        key = cache_key("tts", request.text, self.tts.active_model, voice)
        cached = self.tts_cache.get(key)
        if cached is not None:
            wav, model = cached
            cache_hit = True
        else:
            cache_hit = False
            try:
                result = await self.tts.synthesize(request.text)
            except LLMError as exc:
                log_event(
                    "tts_request",
                    text_chars=len(request.text),
                    model=self.tts.active_model,
                    latency_ms=int((self._clock() - started) * 1000),
                    cache_hit=False,
                    outcome=type(exc).__name__,
                    error=str(exc)[:200],
                )
                raise
            wav, model = result.wav, result.model
            self.tts_cache.set(key, (wav, model))
        latency_ms = int((self._clock() - started) * 1000)
        log_event(
            "tts_request",
            text_chars=len(request.text),
            model=model,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            bytes=len(wav),
            outcome="ok",
        )
        return wav, model, cache_hit

    # --- internals ---------------------------------------------------------------------

    async def _ask(self, context: PageContext, question: str, request_id: str) -> AnswerResponse:
        text, scope, meta = await self._generate(context, "ask", question, request_id)
        return AnswerResponse(answer=text, scope=scope or "unknown", meta=meta)

    async def _summarize(self, context: PageContext, request_id: str) -> SummaryResponse:
        text, _, meta = await self._generate(context, "summarize", None, request_id)
        return SummaryResponse(summary=text, meta=meta)

    async def _generate(
        self, context: PageContext, action: Action, question: str | None, request_id: str
    ) -> tuple[str, Scope | None, Meta]:
        started = self._clock()
        system, user = build_prompt(context, action, question)
        key = cache_key(
            context.page,
            action,
            context.to_prompt_text(),
            context.metadata_text(),
            (question or "").strip().lower(),
            self.llm.active_model,
            PROMPT_VERSION,
        )
        base_fields = dict(
            page=context.page, action=action, rows_total=context.rows_total, prompt_chars=len(system) + len(user)
        )
        scope: Scope | None = None
        input_tokens = output_tokens = None
        outcome = "ok"

        cached = self.cache.get(key)
        if cached is not None:
            text, scope, model = cached
            cache_hit = True
        else:
            cache_hit = False
            try:
                result = await self.llm.generate(
                    system,
                    user,
                    response_schema=AskOutput if action == "ask" else None,
                    **GENERATION_PARAMS[action],
                )
            except LLMError as exc:
                log_event(
                    "ai_request",
                    **base_fields,
                    model=self.llm.active_model,
                    latency_ms=int((self._clock() - started) * 1000),
                    cache_hit=False,
                    outcome=type(exc).__name__,
                    error=str(exc)[:200],
                )
                raise
            model = result.model
            input_tokens, output_tokens = result.input_tokens, result.output_tokens
            if action == "ask":
                text, scope = parse_ask(result.text)
            else:
                text = clean_text(result.text)
            text, screened = screen_output(text)
            if screened:
                outcome = "screened"
            else:
                self.cache.set(key, (text, scope, model))

        latency_ms = int((self._clock() - started) * 1000)
        log_event(
            "ai_request",
            **base_fields,
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=cache_hit,
            scope=scope,
            outcome=outcome,
        )
        return text, scope, Meta(model=model, latency_ms=latency_ms, cached=cache_hit, request_id=request_id)
