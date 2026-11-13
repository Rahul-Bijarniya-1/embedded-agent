"""Per-client token bucket for the AI routes. In-memory: one process, one bucket table."""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request

RATE_LIMITED_BODY = {"error": "rate_limited", "message": "Too many requests; try again in a few seconds."}
_MAX_TRACKED_CLIENTS = 5000


class TokenBucketLimiter:
    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.per_minute = max(0, int(per_minute))
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, client_id: str) -> tuple[bool, float]:
        """(allowed, seconds until the next token) for this client."""
        if self.per_minute == 0:
            return True, 0.0
        now = self._clock()
        rate = self.per_minute / 60.0
        tokens, last = self._buckets.get(client_id, (float(self.per_minute), now))
        tokens = min(float(self.per_minute), tokens + (now - last) * rate)
        if len(self._buckets) > _MAX_TRACKED_CLIENTS:
            self._buckets = {k: v for k, v in self._buckets.items() if now - v[1] < 60}
        if tokens >= 1.0:
            self._buckets[client_id] = (tokens - 1.0, now)
            return True, 0.0
        self._buckets[client_id] = (tokens, now)
        return False, (1.0 - tokens) / rate


def client_id(request: Any) -> str:
    headers = getattr(request, "headers", None) or {}
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "unknown"


class RateLimitDependency:
    """FastAPI dependency: raises 429 with Retry-After when the client's bucket is empty."""

    def __init__(self, limiter: TokenBucketLimiter):
        self.limiter = limiter

    async def __call__(self, request: Request) -> None:
        allowed, retry_after = self.limiter.allow(client_id(request))
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=dict(RATE_LIMITED_BODY),
                headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
            )
