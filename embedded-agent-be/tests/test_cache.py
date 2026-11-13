from datetime import date

from agent.cache import TTLCache, cache_key


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_cache_hit_miss_and_stats():
    cache = TTLCache(maxsize=2, ttl_s=10, clock=Clock())
    assert cache.get("a") is None
    cache.set("a", 1)
    assert cache.get("a") == 1
    assert cache.stats() == {"size": 1, "hits": 1, "misses": 1}


def test_cache_expires_after_ttl():
    clock = Clock()
    cache = TTLCache(maxsize=2, ttl_s=10, clock=clock)
    cache.set("a", 1)
    clock.t += 9.9
    assert cache.get("a") == 1
    clock.t += 0.2
    assert cache.get("a") is None
    assert cache.stats()["size"] == 0


def test_cache_evicts_lru():
    cache = TTLCache(maxsize=2, ttl_s=100, clock=Clock())
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.stats()["size"] == 2


def test_cache_set_overwrites_without_growing():
    cache = TTLCache(maxsize=2, ttl_s=100, clock=Clock())
    cache.set("a", 1)
    cache.set("a", 2)
    assert cache.get("a") == 2
    assert cache.stats()["size"] == 1


def test_cache_key_is_stable_and_order_insensitive():
    k1 = cache_key("transactions", "ask", {"b": 1, "a": 2}, "q")
    k2 = cache_key("transactions", "ask", {"a": 2, "b": 1}, "q")
    assert k1 == k2
    assert len(k1) == 64
    assert cache_key("x") != cache_key("y")
    assert cache_key(date(2026, 1, 1)) == cache_key(date(2026, 1, 1))
