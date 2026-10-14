from app.services.cache import InMemoryTTLCache


def test_cache_reuses_value_until_expiration():
    now = [0.0]
    cache = InMemoryTTLCache(maxsize=2, ttl_seconds=10, timer=lambda: now[0])
    calls = []

    first = cache.get_or_load("movie:heist", lambda: calls.append("load") or ["title"])
    second = cache.get_or_load("movie:heist", lambda: calls.append("load again"))

    assert first == second == ["title"]
    assert calls == ["load"]

    now[0] = 11.0
    refreshed = cache.get_or_load("movie:heist", lambda: calls.append("refresh") or ["new"])

    assert refreshed == ["new"]
    assert calls == ["load", "refresh"]


def test_cache_evicts_oldest_entry_when_full():
    cache = InMemoryTTLCache(maxsize=2, ttl_seconds=60)
    cache.get_or_load("a", lambda: "A")
    cache.get_or_load("b", lambda: "B")
    cache.get_or_load("a", lambda: "unexpected")
    cache.get_or_load("c", lambda: "C")

    assert cache.get_or_load("a", lambda: "unexpected") == "A"
    assert cache.get_or_load("b", lambda: "reloaded") == "reloaded"