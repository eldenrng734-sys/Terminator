from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import main
from app.auth import require_user
from app.recommend import router as recommendation_router
from app.recommend.models import RecommendationIntent, ScoredCandidate
from app.services.session import SessionStore


class FakeTMDBClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    monkeypatch.setattr(main, "session_store", store)
    monkeypatch.setattr(recommendation_router, "session_store", store)
    monkeypatch.setitem(main.app.dependency_overrides, require_user, lambda: 1)
    return TestClient(main.app)


def _intent(mode: str = "discovery", **values) -> RecommendationIntent:
    return RecommendationIntent(mode=mode, **values)


def _candidate() -> ScoredCandidate:
    return ScoredCandidate(
        tmdb_id=101,
        media_type="movie",
        title="A Good Film",
        vote_average=8.0,
        vote_count=1000,
        score=1.0,
    )


def test_off_topic_returns_422_without_tmdb_or_reply_calls(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        recommendation_router,
        "extract_recommendation_intent",
        lambda text, history=None: _intent("off_topic"),
    )
    create_client = Mock(side_effect=AssertionError("TMDB must not be called"))
    write_reply = Mock(side_effect=AssertionError("reply LLM must not be called"))
    monkeypatch.setattr(recommendation_router, "create_client", create_client)
    monkeypatch.setattr(recommendation_router, "_write_reply", write_reply)

    response = client.post("/api/recommend", json={"query": "weather tomorrow"})

    assert response.status_code == 422
    assert "handles movie and TV" in response.json()["detail"]
    create_client.assert_not_called()
    write_reply.assert_not_called()


def test_similarity_without_reference_title_returns_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        recommendation_router,
        "extract_recommendation_intent",
        lambda text, history=None: _intent("similarity", media_type="movie"),
    )
    create_client = Mock(side_effect=AssertionError("TMDB must not be called"))
    monkeypatch.setattr(recommendation_router, "create_client", create_client)

    response = client.post("/api/recommend", json={"query": "something similar"})

    assert response.status_code == 422
    assert "identify the title" in response.json()["detail"]
    create_client.assert_not_called()


def test_last_watched_runs_title_resolution_and_similarity_pool(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        recommendation_router,
        "extract_recommendation_intent",
        lambda text, history=None: _intent(
            "similarity",
            media_type="movie",
            reference_title="Interstellar",
            reference_year=2014,
        ),
    )
    monkeypatch.setattr(recommendation_router, "create_client", FakeTMDBClient)
    resolved = []
    recommended = []

    async def fake_resolve(client, media_type, title, year=None):
        resolved.append((media_type, title, year))
        return {"id": 77}

    async def fake_watched(client, media_type, tmdb_id, limit=5, watch_region=None):
        recommended.append((media_type, tmdb_id, limit, watch_region))
        return [_candidate()]

    async def fake_reply(query, candidates):
        return "A mind-expanding film."

    monkeypatch.setattr(recommendation_router, "resolve_title", fake_resolve)
    monkeypatch.setattr(
        recommendation_router, "recommend_from_watched", fake_watched
    )
    monkeypatch.setattr(recommendation_router, "_write_reply", fake_reply)

    response = client.post(
        "/api/recommend",
        json={"query": "I loved Interstellar, anything similar?", "media_type": "movie"},
    )

    assert response.status_code == 200
    assert resolved == [("movie", "Interstellar", 2014)]
    assert recommended == [("movie", 77, 5, "US")]


def test_unresolved_title_returns_empty_candidate_fallback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        recommendation_router,
        "extract_recommendation_intent",
        lambda text, history=None: _intent(
            "similarity", media_type="movie", reference_title="Unknown Film"
        ),
    )
    monkeypatch.setattr(recommendation_router, "create_client", FakeTMDBClient)

    async def unresolved_title(client, media_type, title, year=None):
        return None

    recommend_watched = Mock(side_effect=AssertionError("unresolved title must be skipped"))
    monkeypatch.setattr(recommendation_router, "resolve_title", unresolved_title)
    monkeypatch.setattr(
        recommendation_router, "recommend_from_watched", recommend_watched
    )

    response = client.post("/api/recommend", json={"query": "similar to Unknown Film"})

    assert response.status_code == 200
    assert response.json()["candidates"] == []
    assert "couldn't find a strong match" in response.json()["reply"]
    recommend_watched.assert_not_called()


def test_vague_request_still_runs_discovery(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        recommendation_router,
        "extract_recommendation_intent",
        lambda text, history=None: _intent("discovery", media_type="movie"),
    )
    monkeypatch.setattr(recommendation_router, "create_client", FakeTMDBClient)

    async def empty_genres(client, media_type, terms):
        return []

    async def empty_keywords(client, terms):
        return []

    async def empty_providers(client, media_type, terms, region):
        return []

    monkeypatch.setattr(recommendation_router, "resolve_genre_ids", empty_genres)
    monkeypatch.setattr(recommendation_router, "resolve_keyword_ids", empty_keywords)
    monkeypatch.setattr(
        recommendation_router, "resolve_watch_provider_ids", empty_providers
    )
    discovered = []

    async def fake_discovery(client, media_type, *, filters, limit):
        discovered.append((media_type, filters, limit))
        return [_candidate()]

    async def fake_reply(query, candidates):
        return "A confident pick."

    monkeypatch.setattr(recommendation_router, "recommend_from_query", fake_discovery)
    monkeypatch.setattr(recommendation_router, "_write_reply", fake_reply)

    response = client.post("/api/recommend", json={"query": "something good"})

    assert response.status_code == 200
    assert response.json()["candidates"][0]["tmdb_id"] == 101
    assert len(discovered) == 1
    media_type, filters, limit = discovered[0]
    assert media_type == "movie"
    assert limit == 5
    assert filters["genre_ids"] == []
    assert filters["keyword_ids_any"] == []


def test_refinement_inherits_prior_filters_and_saves_once(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = main.session_store.new_session(user_id=1)
    main.session_store.append(
        session_id,
        {"role": "user", "content": "Recommend a comedy"},
        "Here are comedies.",
        user_id=1,
    )
    append = Mock(wraps=main.session_store.append)
    monkeypatch.setattr(main.session_store, "append", append)
    monkeypatch.setattr(recommendation_router, "create_client", FakeTMDBClient)
    extraction_calls = []

    def fake_extract(text, history=None):
        extraction_calls.append((text, history))
        if text == "something funnier":
            return _intent("discovery", media_type="movie", mood_terms=["funny"])
        return _intent("discovery", media_type="movie", genre_terms=["comedy"])

    monkeypatch.setattr(
        recommendation_router, "extract_recommendation_intent", fake_extract
    )

    async def genre_ids(client, media_type, terms):
        return [35]

    async def keyword_ids(client, terms):
        return [77]

    async def provider_ids(client, media_type, terms, region):
        return []

    monkeypatch.setattr(recommendation_router, "resolve_genre_ids", genre_ids)
    monkeypatch.setattr(recommendation_router, "resolve_keyword_ids", keyword_ids)
    monkeypatch.setattr(
        recommendation_router, "resolve_watch_provider_ids", provider_ids
    )
    discovered = []

    async def fake_discovery(client, media_type, *, filters, limit):
        discovered.append(filters)
        return [_candidate()]

    async def fake_reply(query, candidates):
        return "A funnier comedy."

    monkeypatch.setattr(recommendation_router, "recommend_from_query", fake_discovery)
    monkeypatch.setattr(recommendation_router, "_write_reply", fake_reply)

    response = client.post(
        "/api/recommend?session_id=" + session_id,
        json={"query": "something funnier"},
    )

    assert response.status_code == 200
    assert discovered[0]["genre_ids"] == [35]
    assert discovered[0]["keyword_ids_any"] == [77]
    assert extraction_calls[0][1][0]["content"] == "Recommend a comedy"
    assert append.call_count == 1


def test_chat_passes_recent_history_and_appends_once(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = main.session_store.new_session(user_id=1)
    main.session_store.append(
        session_id,
        {"role": "user", "content": "Recommend a comedy"},
        "Here are comedies.",
        user_id=1,
    )
    append = Mock(wraps=main.session_store.append)
    monkeypatch.setattr(main.session_store, "append", append)
    seen = {}

    async def fake_run(payload, history=None, surprise=False):
        seen["query"] = payload.query
        seen["history"] = history
        seen["surprise"] = surprise
        return SimpleNamespace(reply="A funnier comedy.")

    monkeypatch.setattr(main, "run_recommendation", fake_run)

    response = client.post(
        "/api/chat",
        data={"message": "something funnier", "session_id": session_id},
    )

    assert response.status_code == 200
    assert seen["query"] == "something funnier"
    assert seen["history"][0]["content"] == "Recommend a comedy"
    assert seen["surprise"] is False
    assert append.call_count == 1