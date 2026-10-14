import json

import pytest

from app.recommend import router as recommendation_router
from app.recommend.models import ScoredCandidate


def _candidate(tmdb_id=1, media_type="movie", title="Title 1", year=2001):
    return ScoredCandidate(
        tmdb_id=tmdb_id,
        media_type=media_type,
        title=title,
        year=year,
        vote_average=8.0,
        vote_count=1000,
        runtime=120,
        original_language="en",
        overview="A great film.",
        reasons=["shares genre"],
        score=1.0,
    )


def _selection_response(selections):
    return json.dumps({"selections": selections})


@pytest.mark.asyncio
async def test_valid_selection_renders_verified_titles(monkeypatch):
    candidates = [_candidate(1, "movie", "Alpha", 2001), _candidate(2, "tv", "Beta", 2005)]
    seen = {}

    def fake_chat_completion(messages, **kwargs):
        seen["messages"] = messages
        seen.update(kwargs)
        return _selection_response(
            [
                {"tmdb_id": 1, "media_type": "movie", "why": "Fits your tense mood."},
                {"tmdb_id": 2, "media_type": "tv", "why": "A lighter second pick."},
            ]
        )

    monkeypatch.setattr(recommendation_router, "chat_completion", fake_chat_completion)

    reply = await recommendation_router._write_reply("tense thriller", candidates)

    assert "Alpha" in reply and "Beta" in reply
    assert "shares genre" in reply
    # candidate list adjacent to user request
    blob = json.dumps(seen["messages"])
    assert "CANDIDATE LIST" in blob
    assert "tense thriller" in blob
    assert seen["temperature"] == 0.0
    assert seen["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_unknown_candidate_id_is_dropped(monkeypatch):
    candidates = [_candidate(1, "movie", "Alpha", 2001)]

    def fake_chat_completion(messages, **kwargs):
        return _selection_response(
            [
                {"tmdb_id": 1, "media_type": "movie", "why": "Verified pick."},
                {"tmdb_id": 999, "media_type": "movie", "why": "Unknown pick."},
            ]
        )

    monkeypatch.setattr(recommendation_router, "chat_completion", fake_chat_completion)

    reply = await recommendation_router._write_reply("something good", candidates)

    assert "Alpha" in reply
    assert "Unknown pick" not in reply
    assert "999" not in reply


@pytest.mark.asyncio
async def test_malformed_model_output_uses_fallback(monkeypatch):
    candidates = [_candidate(1, "movie", "Alpha", 2001)]
    monkeypatch.setattr(
        recommendation_router, "chat_completion", lambda messages, **kw: "not-json{{{"
    )

    reply = await recommendation_router._write_reply("something good", candidates)

    assert "Alpha" in reply  # fallback lists verified candidate
    assert "not-json" not in reply


@pytest.mark.asyncio
async def test_empty_shortlist_returns_empty_message(monkeypatch):
    called = []

    def fake_chat_completion(messages, **kwargs):
        called.append(1)
        return "{}"

    monkeypatch.setattr(recommendation_router, "chat_completion", fake_chat_completion)

    reply = await recommendation_router._write_reply("something good", [])

    assert "couldn't find a strong match" in reply
    assert called == []


@pytest.mark.asyncio
async def test_outside_shortlist_movie_is_never_shown(monkeypatch):
    candidates = [_candidate(1, "movie", "Alpha", 2001)]

    def fake_chat_completion(messages, **kwargs):
        return _selection_response(
            [{"tmdb_id": 7, "media_type": "movie", "why": "Invented Movie is great."}]
        )

    monkeypatch.setattr(recommendation_router, "chat_completion", fake_chat_completion)

    reply = await recommendation_router._write_reply("something good", candidates)

    assert "Invented Movie" not in reply
    assert "Alpha" in reply  # safe fallback to verified shortlist


@pytest.mark.asyncio
async def test_valid_candidate_selection_cannot_inject_unverified_title(monkeypatch):
    candidates = [_candidate(1, "movie", "Alpha", 2001)]

    def fake_chat_completion(messages, **kwargs):
        return _selection_response(
            [
                {
                    "tmdb_id": 1,
                    "media_type": "movie",
                    "why": "It is better than Invented Movie.",
                }
            ]
        )

    monkeypatch.setattr(recommendation_router, "chat_completion", fake_chat_completion)

    reply = await recommendation_router._write_reply("something good", candidates)

    assert "Alpha" in reply
    assert "Invented Movie" not in reply
