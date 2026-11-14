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


def test_client_id_ignores_forwarded_for_unless_trusted():
    behind_proxy = SimpleNamespace(headers={"x-forwarded-for": "9.9.9.9, 10.0.0.1"}, client=SimpleNamespace(host="1.1.1.1"))
    assert client_id(behind_proxy) == "1.1.1.1"
    assert client_id(behind_proxy, trust_forwarded=True) == "9.9.9.9"
    assert client_id(SimpleNamespace(headers={}, client=SimpleNamespace(host="1.1.1.1")), trust_forwarded=True) == "1.1.1.1"
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


def _request(forwarded, host="1.2.3.4"):
    return SimpleNamespace(headers={"x-forwarded-for": forwarded}, client=SimpleNamespace(host=host))


async def test_a_spoofed_forwarded_header_does_not_buy_a_fresh_bucket():
    dependency = RateLimitDependency(TokenBucketLimiter(per_minute=1, clock=Clock()))
    await dependency(_request("10.0.0.1"))
    with pytest.raises(HTTPException) as info:
        await dependency(_request("10.0.0.2"))
    assert info.value.status_code == 429


async def test_behind_a_trusted_proxy_the_forwarded_address_separates_clients():
    dependency = RateLimitDependency(TokenBucketLimiter(per_minute=1, clock=Clock()), trust_forwarded=True)
    await dependency(_request("10.0.0.1"))
    await dependency(_request("10.0.0.2"))
    with pytest.raises(HTTPException):
        await dependency(_request("10.0.0.1"))


async def test_the_router_dependency_uses_the_trust_setting(monkeypatch):
    from agent import router as router_module
    from agent.config import Settings

    monkeypatch.setattr(router_module, "get_settings", lambda: Settings(_env_file=None, rate_limit_per_minute=1))
    monkeypatch.setitem(router_module._state, "limiter", None)
    router_module._state.pop("limiter")
    try:
        await router_module.rate_limit_dependency(_request("10.0.0.1"))
        with pytest.raises(HTTPException):
            await router_module.rate_limit_dependency(_request("10.0.0.2"))
    finally:
        router_module._state.pop("limiter", None)
