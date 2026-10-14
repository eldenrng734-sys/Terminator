from types import SimpleNamespace

import pytest

from app.recommend import discovery, enrich, similarity
from app.recommend.tmdb_client import TMDBError


def _raw(tmdb_id=1, title="Film"):
    return {
        "id": tmdb_id,
        "title": title,
        "vote_average": 8.0,
        "vote_count": 500,
        "credits": {"cast": []},
        "keywords": {"keywords": []},
        "external_ids": {},
        "release_date": "2020-01-01",
        "genres": [],
        "genre_ids": [],
    }


def _provider_data(results):
    return {"results": results}


@pytest.mark.asyncio
async def test_movie_flatrate_rent_buy_deduped(monkeypatch):
    async def fake_providers(client, media_type, tmdb_id):
        assert (media_type, tmdb_id) == ("movie", 1)
        return _provider_data(
            {
                "US": {
                    "link": "https://tmdb/us/1",
                    "flatrate": [
                        {"provider_id": 8, "provider_name": "Netflix"},
                        {"provider_id": 8, "provider_name": "Netflix"},
                    ],
                    "rent": [{"provider_id": 2, "provider_name": "Apple TV"}],
                    "buy": [{"provider_id": 2, "provider_name": "Apple TV"}],
                }
            }
        )

    monkeypatch.setattr(enrich, "get_watch_providers", fake_providers)

    candidate = await enrich.to_candidate(None, "movie", _raw(1), watch_region="US")

    assert candidate.watch_providers == ["Netflix", "Apple TV"]
    assert candidate.watch_providers_by_category == {
        "flatrate": ["Netflix"],
        "rent": ["Apple TV"],
        "buy": ["Apple TV"],
    }
    assert candidate.watch_link == "https://tmdb/us/1"
    assert candidate.watch_region == "US"


@pytest.mark.asyncio
async def test_tv_free_ads_categories(monkeypatch):
    async def fake_providers(client, media_type, tmdb_id):
        return _provider_data(
            {
                "US": {
                    "link": "https://tmdb/us/2",
                    "free": [{"provider_id": 10, "provider_name": "Tubi"}],
                    "ads": [{"provider_id": 11, "provider_name": "Pluto TV"}],
                }
            }
        )

    monkeypatch.setattr(enrich, "get_watch_providers", fake_providers)

    candidate = await enrich.to_candidate(None, "tv", _raw(2, "Show"), watch_region="us")

    assert candidate.watch_providers == ["Tubi", "Pluto TV"]
    assert candidate.watch_providers_by_category["free"] == ["Tubi"]
    assert candidate.watch_providers_by_category["ads"] == ["Pluto TV"]
    assert candidate.watch_link == "https://tmdb/us/2"


@pytest.mark.asyncio
async def test_unknown_region_falls_back_to_us(monkeypatch):
    async def fake_providers(client, media_type, tmdb_id):
        return _provider_data(
            {"US": {"link": "https://tmdb/us/1", "flatrate": [{"provider_id": 8, "provider_name": "Netflix"}]}}
        )

    monkeypatch.setattr(enrich, "get_watch_providers", fake_providers)

    candidate = await enrich.to_candidate(None, "movie", _raw(1), watch_region="XX")

    assert candidate.watch_providers == ["Netflix"]
    assert candidate.watch_region == "US"
    assert candidate.watch_link == "https://tmdb/us/1"


@pytest.mark.asyncio
async def test_missing_region_returns_consistent_empty_shape(monkeypatch):
    async def fake_providers(client, media_type, tmdb_id):
        return {"results": {}}

    monkeypatch.setattr(enrich, "get_watch_providers", fake_providers)

    candidate = await enrich.to_candidate(None, "movie", _raw(1), watch_region=None)

    assert candidate.watch_providers == []
    assert candidate.watch_providers_by_category == {}
    assert candidate.watch_link is None
    assert candidate.watch_region == "US"
    assert "watch_providers" in candidate.model_dump()


@pytest.mark.asyncio
async def test_provider_error_returns_candidate_with_empty_providers(monkeypatch):
    async def failing(client, media_type, tmdb_id):
        raise TMDBError("timeout")

    monkeypatch.setattr(enrich, "get_watch_providers", failing)

    candidate = await enrich.to_candidate(None, "movie", _raw(1), watch_region="US")

    assert candidate.title == "Film"
    assert candidate.watch_providers == []
    assert candidate.watch_providers_by_category == {}
    assert candidate.watch_link is None


@pytest.mark.asyncio
async def test_both_paths_attach_providers_and_memoize(monkeypatch):
    monkeypatch.setattr(
        enrich, "get_settings", lambda: SimpleNamespace(omdb_api_key=None)
    )
    calls: list[tuple[str, int]] = []

    async def fake_providers(client, media_type, tmdb_id):
        calls.append((media_type, tmdb_id))
        return _provider_data(
            {"US": {"link": f"https://tmdb/{tmdb_id}", "flatrate": [{"provider_id": 8, "provider_name": "Netflix"}]}}
        )

    async def fake_details(client, media_type, tmdb_id):
        return _raw(tmdb_id, f"Title {tmdb_id}")

    monkeypatch.setattr(enrich, "get_watch_providers", fake_providers)
    monkeypatch.setattr(enrich, "get_details", fake_details)
    monkeypatch.setattr(similarity, "get_details", fake_details)

    async def fake_similar(client, media_type, tmdb_id):
        return [{"id": 2}]

    async def fake_recs(client, media_type, tmdb_id):
        return [{"id": 3}]

    monkeypatch.setattr(similarity, "get_similar", fake_similar)
    monkeypatch.setattr(similarity, "get_recommendations", fake_recs)

    watched = await similarity.recommend_from_watched(
        None, "movie", 1, limit=5, watch_region="US"
    )
    assert watched
    assert all(c.watch_providers == ["Netflix"] for c in watched)
    assert all(c.watch_region == "US" for c in watched)

    async def fake_discover(client, media_type, **params):
        return [_raw(11, "Disco"), _raw(12, "Disco 2")]

    monkeypatch.setattr(discovery, "discover", fake_discover)
    found = await discovery.recommend_from_query(
        None, "movie", filters={"watch_region": "US"}, limit=5
    )
    assert found
    assert all(c.watch_providers == ["Netflix"] for c in found)

    # per-request memo: same candidate twice in one request fetches once
    calls.clear()
    memo: enrich.WatchMemo = {}
    raw = _raw(21, "Memo")
    first = await enrich.to_candidate(None, "movie", raw, watch_region="US", watch_memo=memo)
    second = await enrich.to_candidate(None, "movie", raw, watch_region="US", watch_memo=memo)
    assert first.watch_providers == second.watch_providers == ["Netflix"]
    assert len(calls) == 1
