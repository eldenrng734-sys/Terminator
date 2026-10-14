from collections.abc import Callable, Hashable
from asyncio import Lock
from threading import Condition, RLock
from time import monotonic
from typing import Any, TypeVar

from cachetools import TTLCache

T = TypeVar("T")


class InMemoryTTLCache:
    def __init__(
        self,
        maxsize: int,
        ttl_seconds: float,
        timer: Callable[[], float] = monotonic,
    ) -> None:
        self._cache: TTLCache = TTLCache(
            maxsize=maxsize, ttl=ttl_seconds, timer=timer
        )
        self._condition = Condition(RLock())
        self._pending: set[Hashable] = set()

    def get_or_load(self, key: Hashable, loader: Callable[[], T]) -> T:
        with self._condition:
            while True:
                try:
                    return self._cache[key]
                except KeyError:
                    if key not in self._pending:
                        self._pending.add(key)
                        break
                    self._condition.wait()

        try:
            value = loader()
        except BaseException:
            with self._condition:
                self._pending.discard(key)
                self._condition.notify_all()
            raise

        with self._condition:
            self._cache[key] = value
            self._pending.discard(key)
            self._condition.notify_all()
        return value

    def clear(self) -> None:
        with self._condition:
            self._cache.clear()


class AsyncInMemoryTTLCache:
    def __init__(self, maxsize: int, ttl_seconds: float) -> None:
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        self._lock = Lock()

    async def get_or_load(self, key: Hashable, loader: Callable[[], Any]) -> Any:
        async with self._lock:
            try:
                return self._cache[key]
            except KeyError:
                pass

        value = await loader()
        async with self._lock:
            try:
                return self._cache[key]
            except KeyError:
                self._cache[key] = value
                return value

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()