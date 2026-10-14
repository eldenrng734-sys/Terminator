import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from app.auth import require_user
from app.recommend import discovery, enrich, intent, keywords, similarity, tmdb_client
from app.recommend.models import (
    Candidate,
    RecommendationIntent,
    ScoredCandidate,
)
from app.recommend import router as recommendation_router
from app.recommend.scoring import filter_and_rank


def _candidate(
    tmdb_id: int,
    *,
    genre_ids: list[int] | None = None,
    keyword_ids: list[int] | None = None,
    cast_ids: list[int] | None = None,
    vote_average: float = 8.0,
    vote_count: int = 1000,
) -> Candidate:
    return Candidate(
        tmdb_id=tmdb_id,
        media_type="movie",
        title=f"Title {tmdb_id}",
        genre_ids=genre_ids or [],
        keyword_ids=keyword_ids or [],
        cast_ids=cast_ids or [],
        vote_average=vote_average,
        vote_count=vote_count,
    )


def test_quality_filter_and_similarity_scoring_use_shared_metadata():
    source = _candidate(1, genre_ids=[18], keyword_ids=[4], cast_ids=[7])
    matching = _candidate(2, genre_ids=[18], keyword_ids=[4], cast_ids=[7])
    weak = _candidate(3, vote_average=6.0)

    results = filter_and_rank([source, matching, weak], source=source)

    assert [result.tmdb_id for result in results] == [2]
    assert results[0].reasons == ["shares genre", "similar themes", "shares cast"]


@pytest.mark.asyncio
async def test_poor_source_does_not_bypass_quality_bar(monkeypatch):
    monkeypatch.setattr(
        enrich,
        "get_settings",
        lambda: SimpleNamespace(omdb_api_key=None),
    )
    source = _candidate(1, vote_average=4.0, vote_count=10).model_copy(
        update={"rt_score": 20}
    )
    good = _candidate(2)
    low_votes = _candidate(3, vote_count=199)
    low_average = _candidate(4, vote_average=6.9)
    low_rt = _candidate(5).model_copy(update={"rt_score": 69})
    pool = [source, good, low_votes, low_average, low_rt]

    ranked = filter_and_rank(pool, source=source)
    enriched_ranked = await enrich.rank_with_optional_omdb(
        pool, source=source, limit=5
    )

    for results in (ranked, enriched_ranked):
        assert [candidate.tmdb_id for candidate in results] == [2]
        assert all(candidate.vote_count >= 200 for candidate in results)
        assert all(candidate.vote_average >= 7.0 for candidate in results)
        assert all(
            candidate.rt_score is None or candidate.rt_score >= 70
            for candidate in results
        )


def test_title_match_prefers_exact_year_and_rejects_unrelated_titles():
    results = [
        {"id": 1, "title": "The Thing", "release_date": "1982-06-25"},
        {"id": 2, "title": "The Thing", "release_date": "2011-10-14"},
        {"id": 3, "title": "Something Else", "release_date": "1982-01-01"},
    ]

    match = similarity.select_title_match(results, "The Thing", 1982)
    rejected = similarity.select_title_match(results, "Unrelated Movie")

    assert match is not None and match["id"] == 1
    assert rejected is None


def test_intent_extraction_uses_temperature_zero(monkeypatch):
    seen: dict[str, object] = {}

    def fake_chat_completion(messages, **kwargs):
        seen["messages"] = messages
        seen.update(kwargs)
        return json.dumps(
            {
                "mode": "similarity",
                "media_type": "movie",
                "reference_title": "Interstellar",
                "reference_year": 2014,
                "genre_terms": [],
                "mood_terms": [],
            }
        )

    monkeypatch.setattr(intent, "chat_completion", fake_chat_completion)

    parsed = intent.extract_recommendation_intent("I loved Interstellar. Similar films?")

    assert parsed.reference_title == "Interstellar"
    assert parsed.reference_year == 2014
    assert seen["temperature"] == 0.0


@pytest.mark.asyncio
async def test_genre_and_keyword_terms_resolve_through_tmdb(monkeypatch):
    async def fake_genres(client, media_type):
        return [
            {"id": 53, "name": "Thriller"},
            {"id": 18, "name": "Drama"},
        ]

    async def fake_search_keyword(client, query):
        return [{"id": len(query), "name": query}]

    monkeypatch.setattr(keywords, "get_genres", fake_genres)
    monkeypatch.setattr(keywords, "search_keyword", fake_search_keyword)

    genres = await keywords.resolve_genre_ids(None, "movie", ["psychological thriller"])
    keyword_ids = await keywords.resolve_keyword_ids(None, ["tense"])

    assert genres == [53]
    assert keyword_ids
    assert len(keyword_ids) == len(set(keyword_ids))


@pytest.mark.asyncio
async def test_watch_provider_lookup_requires_region_and_maps_service_name(monkeypatch):
    calls = []

    async def fake_provider_list(client, media_type, watch_region):
        calls.append((media_type, watch_region))
        return [
            {"provider_id": 8, "provider_name": "Netflix"},
            {"provider_id": 9, "provider_name": "Amazon Prime Video"},
        ]

    monkeypatch.setattr(keywords, "get_watch_provider_list", fake_provider_list)

    assert await keywords.resolve_watch_provider_ids(
        None, "movie", ["Netflix"], None
    ) == []
    provider_ids = await keywords.resolve_watch_provider_ids(
        None, "movie", ["Netflix", "Prime Video"], "bd"
    )

    assert provider_ids == [8, 9]
    assert calls == [("movie", "bd")]


@pytest.mark.asyncio
async def test_discovery_relaxes_keywords_before_vote_count(monkeypatch):
    observed: list[dict[str, object]] = []

    async def fake_discover(client, media_type, **params):
        observed.append(params)
        return []

    monkeypatch.setattr(discovery, "discover", fake_discover)

    results = await discovery.recommend_from_query(
        None,
        "movie",
        filters={
            "genre_ids": [53],
            "keyword_ids_any": [101, 202],
            "min_vote_count": 200,
        },
    )

    assert results == []
    assert observed[0]["with_keywords"] == "101|202"
    assert observed[0]["vote_count.gte"] == 200
    assert "with_keywords" not in observed[1]
    assert observed[1]["vote_count.gte"] == 200
    assert "with_keywords" not in observed[2]
    assert observed[2]["vote_count.gte"] == 50


@pytest.mark.asyncio
async def test_similarity_deduplicates_candidate_pool(monkeypatch):
    async def fake_get_details(client, media_type, tmdb_id):
        return {"id": tmdb_id}

    async def fake_get_similar(client, media_type, tmdb_id):
        return [{"id": 2}, {"id": 3}]

    async def fake_get_recommendations(client, media_type, tmdb_id):
        return [{"id": 3}, {"id": 4}]

    async def fake_to_candidate(client, media_type, raw, watch_region=None, **kwargs):
        return _candidate(raw["id"])

    async def fake_rank(pool, source, limit):
        assert source.tmdb_id == 1
        assert {candidate.tmdb_id for candidate in pool} == {2, 3, 4}
        return []

    monkeypatch.setattr(similarity, "get_details", fake_get_details)
    monkeypatch.setattr(similarity, "get_similar", fake_get_similar)
    monkeypatch.setattr(similarity, "get_recommendations", fake_get_recommendations)
    monkeypatch.setattr(similarity, "to_candidate", fake_to_candidate)
    monkeypatch.setattr(similarity, "rank_with_optional_omdb", fake_rank)

    assert await similarity.recommend_from_watched(None, "movie", 1) == []


@pytest.mark.asyncio
async def test_tmdb_client_caches_repeated_discover_request(monkeypatch):
    monkeypatch.setattr(
        tmdb_client,
        "get_settings",
        lambda: SimpleNamespace(tmdb_read_access_token="read-token", tmdb_api_key=None),
    )
    await tmdb_client._response_cache.clear()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"results": [{"id": 11}]}, request=request)

    async with httpx.AsyncClient(
        base_url=tmdb_client.TMDB_BASE_URL,
        headers={"Authorization": "Bearer read-token"},
        transport=httpx.MockTransport(handler),
    ) as client:
        first = await tmdb_client.discover(client, "movie", page=1)
        second = await tmdb_client.discover(client, "movie", page=1)

    assert first == second == [{"id": 11}]
    assert len(requests) == 1
    assert requests[0].headers["Authorization"] == "Bearer read-token"
    await tmdb_client._response_cache.clear()


@pytest.mark.asyncio
async def test_genre_endpoint_parses_genres_field(monkeypatch):
    async def fake_get(client, path, **params):
        assert path == "/genre/movie/list"
        return {"genres": [{"id": 18, "name": "Drama"}]}

    monkeypatch.setattr(tmdb_client, "_get", fake_get)

    assert await tmdb_client.get_genres(None, "movie") == [
        {"id": 18, "name": "Drama"}
    ]


def test_discovery_results_carry_empty_reasons_key():
    pool = [_candidate(2, genre_ids=[18]), _candidate(3, genre_ids=[99])]

    results = filter_and_rank(pool, source=None)

    assert results
    for item in results:
        assert item.reasons == []
        assert "reasons" in item.model_dump()


@pytest.mark.asyncio
async def test_discovery_rank_preserves_empty_reasons_through_double_rank(monkeypatch):
    monkeypatch.setattr(
        enrich,
        "get_settings",
        lambda: SimpleNamespace(omdb_api_key=None),
    )
    pool = [_candidate(2, genre_ids=[18]), _candidate(3, genre_ids=[99])]

    results = await enrich.rank_with_optional_omdb(pool, source=None, limit=5)

    assert results
    for item in results:
        assert item.reasons == []


def test_recommend_api_includes_reasons_for_no_overlap_candidate(monkeypatch):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    candidate = ScoredCandidate(**_candidate(43).model_dump(), score=1.0, reasons=[])

    async def fake_genres(client, media_type, terms):
        return []

    async def fake_keywords(client, terms):
        return []

    async def fake_providers(client, media_type, terms, region):
        return []

    async def fake_discovery(client, media_type, *, filters, limit):
        return [candidate]

    async def fake_reply(query, candidates):
        return "A confident pick."

    monkeypatch.setitem(main.app.dependency_overrides, require_user, lambda: 1)
    monkeypatch.setattr(recommendation_router, "create_client", FakeClient)
    monkeypatch.setattr(
        recommendation_router,
        "extract_recommendation_intent",
        lambda query, history=None: RecommendationIntent(
            mode="discovery", media_type="movie"
        ),
    )
    monkeypatch.setattr(recommendation_router, "resolve_genre_ids", fake_genres)
    monkeypatch.setattr(recommendation_router, "resolve_keyword_ids", fake_keywords)
    monkeypatch.setattr(
        recommendation_router, "resolve_watch_provider_ids", fake_providers
    )
    monkeypatch.setattr(recommendation_router, "recommend_from_query", fake_discovery)
    monkeypatch.setattr(recommendation_router, "_write_reply", fake_reply)

    with TestClient(main.app) as client:
        response = client.post("/api/recommend", json={"query": "something good"})

    assert response.status_code == 200
    payload = response.json()["candidates"][0]
    assert "reasons" in payload
    assert payload["reasons"] == []
    for field in (
        "score",
        "tmdb_id",
        "media_type",
        "title",
        "vote_average",
        "vote_count",
    ):
        assert field in payload


def test_recommend_api_preserves_similarity_reasons_and_fields(monkeypatch):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    candidate = ScoredCandidate(
        **_candidate(44).model_dump(),
        score=4.5,
        reasons=["shares genre", "similar themes"],
    )

    async def fake_watched(client, media_type, tmdb_id, limit=5, watch_region=None):
        return [candidate]

    async def fake_reply(query, candidates):
        return "Similar pick."

    monkeypatch.setitem(main.app.dependency_overrides, require_user, lambda: 1)
    monkeypatch.setattr(recommendation_router, "create_client", FakeClient)
    monkeypatch.setattr(
        recommendation_router,
        "extract_recommendation_intent",
        lambda query, history=None: RecommendationIntent(
            mode="similarity", media_type="movie", reference_title="Interstellar"
        ),
    )
    async def fake_resolve_title(client, media_type, title, year=None):
        return {"id": 1}

    monkeypatch.setattr(recommendation_router, "resolve_title", fake_resolve_title)
    monkeypatch.setattr(recommendation_router, "recommend_from_watched", fake_watched)
    monkeypatch.setattr(recommendation_router, "_write_reply", fake_reply)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/recommend", json={"query": "like Interstellar"}
        )

    assert response.status_code == 200
    payload = response.json()["candidates"][0]
    assert payload["reasons"] == ["shares genre", "similar themes"]
    for field in (
        "score",
        "tmdb_id",
        "media_type",
        "title",
        "vote_average",
        "vote_count",
    ):
        assert field in payload


def test_recommend_route_requires_authentication():
    with TestClient(main.app) as client:
        response = client.post("/api/recommend", json={"query": "a tense thriller"})

    assert response.status_code == 401


def test_recommend_route_returns_candidates_for_authenticated_user(monkeypatch):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    candidate = ScoredCandidate(**_candidate(42).model_dump(), score=1.2, reasons=[])

    async def fake_genres(client, media_type, terms):
        return [53]

    async def fake_keywords(client, terms):
        return []

    async def fake_discovery(client, media_type, *, filters, limit):
        assert media_type == "movie"
        assert filters["genre_ids"] == [53]
        return [candidate]

    async def fake_reply(query, candidates):
        return "Start with this thriller."

    monkeypatch.setitem(main.app.dependency_overrides, require_user, lambda: 1)
    monkeypatch.setattr(recommendation_router, "create_client", FakeClient)
    monkeypatch.setattr(
        recommendation_router,
        "extract_recommendation_intent",
        lambda query, history=None: RecommendationIntent(
            mode="discovery", media_type="movie", genre_terms=["thriller"]
        ),
    )
    monkeypatch.setattr(recommendation_router, "resolve_genre_ids", fake_genres)
    monkeypatch.setattr(recommendation_router, "resolve_keyword_ids", fake_keywords)
    monkeypatch.setattr(recommendation_router, "recommend_from_query", fake_discovery)
    monkeypatch.setattr(recommendation_router, "_write_reply", fake_reply)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/recommend", json={"query": "a tense thriller", "limit": 1}
        )

    assert response.status_code == 200
    assert response.json()["mode"] == "discovery"
    assert response.json()["reply"] == "Start with this thriller."
    assert response.json()["candidates"][0]["tmdb_id"] == 42


def _surprise_candidate(tmdb_id=90):
    return ScoredCandidate(**_candidate(tmdb_id).model_dump(), score=1.5, reasons=[])


def test_surprise_skips_intent_extraction(monkeypatch):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    def forbidden_intent(query, history=None):
        raise AssertionError("intent must not be called for surprise")

    async def fake_discovery(client, media_type, *, filters, limit):
        return [_surprise_candidate(90 if media_type == "movie" else 91)]

    async def fake_reply(query, candidates, history=None):
        return "Surprise pick."

    monkeypatch.setitem(main.app.dependency_overrides, require_user, lambda: 1)
    monkeypatch.setattr(recommendation_router, "create_client", FakeClient)
    monkeypatch.setattr(
        recommendation_router, "extract_recommendation_intent", forbidden_intent
    )
    monkeypatch.setattr(
        recommendation_router, "recommend_from_query", fake_discovery
    )
    monkeypatch.setattr(recommendation_router, "_write_reply", fake_reply)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/recommend?surprise=true", json={"query": "anything"}
        )

    assert response.status_code == 200
    assert response.json()["mode"] == "discovery"


def test_surprise_uses_broad_quality_filters(monkeypatch):
    from app.config import get_settings
    from app.recommend.scoring import MIN_VOTE_COUNT

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    seen = []

    async def fake_discovery(client, media_type, *, filters, limit):
        seen.append((media_type, dict(filters), limit))
        return []

    async def fake_reply(query, candidates, history=None):
        return "empty"

    monkeypatch.setitem(main.app.dependency_overrides, require_user, lambda: 1)
    monkeypatch.setattr(recommendation_router, "create_client", FakeClient)
    monkeypatch.setattr(
        recommendation_router,
        "extract_recommendation_intent",
        lambda query, history=None: (_ for _ in ()).throw(
            AssertionError("intent must not be called")
        ),
    )
    monkeypatch.setattr(
        recommendation_router, "recommend_from_query", fake_discovery
    )
    monkeypatch.setattr(recommendation_router, "_write_reply", fake_reply)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/recommend?surprise=true",
            json={"query": "surprise me", "limit": 3},
        )

    assert response.status_code == 200
    assert seen
    settings = get_settings()
    for _, filters, limit in seen:
        assert filters["genre_ids"] == []
        assert filters["keyword_ids_any"] == []
        assert filters["watch_provider_ids"] == []
        assert filters["min_vote_average"] == settings.surprise_min_vote_average
        assert filters["min_vote_average"] >= 7.0
        assert filters["min_vote_count"] == MIN_VOTE_COUNT == 200
        assert filters["watch_region"] == "US"
        assert limit == 3


def test_surprise_quality_bar_still_filters():
    from app.recommend.scoring import passes_quality_bar

    good = _candidate(91)
    assert passes_quality_bar(good)
    assert not passes_quality_bar(_candidate(92, vote_average=6.9))
    assert not passes_quality_bar(_candidate(93, vote_count=10))


def test_surprise_false_follows_normal_path(monkeypatch):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    intent_calls = []

    def fake_intent(query, history=None):
        intent_calls.append(query)
        return RecommendationIntent(
            mode="discovery", media_type="movie", genre_terms=["thriller"]
        )

    async def fake_discovery(client, media_type, *, filters, limit):
        return [_surprise_candidate()]

    async def fake_reply(query, candidates, history=None):
        return "Normal pick."

    async def fake_genres(client, media_type, terms):
        return [53]

    async def fake_keywords(client, terms):
        return []

    async def fake_providers(client, media_type, terms, region):
        return []

    monkeypatch.setitem(main.app.dependency_overrides, require_user, lambda: 1)
    monkeypatch.setattr(recommendation_router, "create_client", FakeClient)
    monkeypatch.setattr(
        recommendation_router, "extract_recommendation_intent", fake_intent
    )
    monkeypatch.setattr(recommendation_router, "resolve_genre_ids", fake_genres)
    monkeypatch.setattr(recommendation_router, "resolve_keyword_ids", fake_keywords)
    monkeypatch.setattr(
        recommendation_router, "resolve_watch_provider_ids", fake_providers
    )
    monkeypatch.setattr(
        recommendation_router, "recommend_from_query", fake_discovery
    )
    monkeypatch.setattr(recommendation_router, "_write_reply", fake_reply)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/recommend?surprise=false", json={"query": "a tense thriller"}
        )

    assert response.status_code == 200
    assert intent_calls == ["a tense thriller"]


def test_surprise_response_shape_validates(monkeypatch):
    from app.recommend.models import RecommendationResponse

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    async def fake_discovery(client, media_type, *, filters, limit):
        return [_surprise_candidate()]

    async def fake_reply(query, candidates, history=None):
        return "Surprise!"

    monkeypatch.setitem(main.app.dependency_overrides, require_user, lambda: 1)
    monkeypatch.setattr(recommendation_router, "create_client", FakeClient)
    monkeypatch.setattr(
        recommendation_router,
        "extract_recommendation_intent",
        lambda query, history=None: (_ for _ in ()).throw(
            AssertionError("intent must not be called")
        ),
    )
    monkeypatch.setattr(
        recommendation_router, "recommend_from_query", fake_discovery
    )
    monkeypatch.setattr(recommendation_router, "_write_reply", fake_reply)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/recommend?surprise=true", json={"query": "surprise me"}
        )

    assert response.status_code == 200
    parsed = RecommendationResponse.model_validate(response.json())
    assert parsed.mode == "discovery"
    assert isinstance(parsed.reply, str)
    first = parsed.candidates[0]
    assert first.tmdb_id == 90
    assert first.media_type == "movie"
    assert isinstance(first.score, float)
    assert first.reasons == []