"""Small in-memory LRU cache with a TTL. Production would swap this for Redis with the same key."""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


class TTLCache:
    def __init__(self, maxsize: int, ttl_s: float, clock: Callable[[], float] = time.monotonic):
        self.maxsize = max(1, int(maxsize))
        self.ttl_s = float(ttl_s)
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        item = self._items.get(key)
        if item is None:
            self.misses += 1
            return None
        expires_at, value = item
        if self._clock() >= expires_at:
            del self._items[key]
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        self._items[key] = (self._clock() + self.ttl_s, value)
        self._items.move_to_end(key)
        while len(self._items) > self.maxsize:
            self._items.popitem(last=False)

    def stats(self) -> dict[str, int]:
        return {"size": len(self._items), "hits": self.hits, "misses": self.misses}


def cache_key(*parts: Any) -> str:
    """SHA-256 over a canonical JSON encoding of the parts (dict key order does not matter)."""
    encoded = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
