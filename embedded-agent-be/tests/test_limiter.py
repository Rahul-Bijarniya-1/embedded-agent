from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from agent.limiter import RateLimitDependency, TokenBucketLimiter, client_id


class Clock:
    def __init__(self):
        self.t = 50.0

    def __call__(self):
        return self.t


def test_limiter_allows_then_blocks_then_refills():
    clock = Clock()
    limiter = TokenBucketLimiter(per_minute=20, clock=clock)
    assert all(limiter.allow("ip")[0] for _ in range(20))
    allowed, retry_after = limiter.allow("ip")
    assert allowed is False
    assert 0 < retry_after <= 3.0
    clock.t += 3.0
    assert limiter.allow("ip")[0] is True
    assert limiter.allow("ip")[0] is False


def test_limiter_zero_disables():
    limiter = TokenBucketLimiter(per_minute=0, clock=Clock())
    assert all(limiter.allow("ip") == (True, 0.0) for _ in range(100))


def test_limiter_isolates_clients():
    limiter = TokenBucketLimiter(per_minute=1, clock=Clock())
    assert limiter.allow("a")[0] is True
    assert limiter.allow("a")[0] is False
    assert limiter.allow("b")[0] is True


def test_client_id_prefers_forwarded_for():
    assert client_id(SimpleNamespace(headers={"x-forwarded-for": "9.9.9.9, 10.0.0.1"}, client=SimpleNamespace(host="1.1.1.1"))) == "9.9.9.9"
    assert client_id(SimpleNamespace(headers={}, client=SimpleNamespace(host="1.1.1.1"))) == "1.1.1.1"
    assert client_id(SimpleNamespace(headers={}, client=None)) == "unknown"


async def test_rate_limit_dependency_raises_429_with_retry_after():
    dependency = RateLimitDependency(TokenBucketLimiter(per_minute=1, clock=Clock()))
    request = SimpleNamespace(headers={}, client=SimpleNamespace(host="1.2.3.4"))
    await dependency(request)
    with pytest.raises(HTTPException) as info:
        await dependency(request)
    assert info.value.status_code == 429
    assert info.value.detail == {"error": "rate_limited", "message": "Too many requests; try again in a few seconds."}
    assert int(info.value.headers["Retry-After"]) >= 1
