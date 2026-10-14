import json

import pytest

from app.recommend import intent
from app.recommend.intent import RecommendationIntentError


def _json_response(**overrides):
    payload = {
        "mode": "discovery",
        "media_type": "both",
        "reference_title": None,
        "reference_year": None,
        "genre_terms": [],
        "mood_terms": [],
        "year_from": None,
        "year_to": None,
        "runtime_min": None,
        "runtime_max": None,
        "watch_provider_terms": [],
        "language": None,
        "watch_region": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


def _mock_classifier(monkeypatch, response):
    captured = {}

    def fake_chat_completion(messages, **kwargs):
        captured["messages"] = messages
        captured.update(kwargs)
        return response

    monkeypatch.setattr(intent, "chat_completion", fake_chat_completion)
    return captured


def test_last_watched_classifies_liked_title_as_similarity(monkeypatch):
    _mock_classifier(
        monkeypatch,
        _json_response(
            mode="similarity",
            media_type="movie",
            reference_title="Interstellar",
            reference_year=2014,
        ),
    )

    result = intent.extract_recommendation_intent(
        "I loved Interstellar, anything similar?"
    )

    assert result.mode == "similarity"
    assert result.reference_title == "Interstellar"
    assert result.reference_year == 2014


def test_genre_mood_request_extracts_terms(monkeypatch):
    _mock_classifier(
        monkeypatch,
        _json_response(
            genre_terms=["science fiction", "thriller"],
            mood_terms=["tense"],
        ),
    )

    result = intent.extract_recommendation_intent(
        "Recommend a tense sci-fi thriller"
    )

    assert result.mode == "discovery"
    assert result.genre_terms == ["science fiction", "thriller"]
    assert result.mood_terms == ["tense"]


def test_vague_request_without_history_stays_empty_discovery(monkeypatch):
    captured = _mock_classifier(monkeypatch, _json_response())

    result = intent.extract_recommendation_intent("something good")

    assert result.mode == "discovery"
    assert result.media_type == "both"
    assert result.genre_terms == []
    assert result.mood_terms == []
    assert len(captured["messages"]) == 2
    assert captured["messages"][-1] == {
        "role": "user",
        "content": "something good",
    }


def test_off_topic_request_classifies_as_off_topic(monkeypatch):
    _mock_classifier(monkeypatch, _json_response(mode="off_topic"))

    result = intent.extract_recommendation_intent("What's the weather tomorrow?")

    assert result.mode == "off_topic"


def test_refinement_receives_history_and_requests_json_at_zero_temperature(monkeypatch):
    history = [
        {"role": "user", "content": "Recommend a comedy."},
        {"role": "assistant", "content": "Here are some comedies."},
    ]
    captured = _mock_classifier(
        monkeypatch,
        _json_response(genre_terms=["comedy"], mood_terms=["funny"]),
    )

    result = intent.extract_recommendation_intent("something funnier", history)

    assert result.mode == "discovery"
    assert result.reference_title is None
    assert result.genre_terms == ["comedy"]
    assert result.mood_terms == ["funny"]
    assert len(captured["messages"]) == 3
    assert "Recommend a comedy" in captured["messages"][1]["content"]
    assert "Here are some comedies" in captured["messages"][1]["content"]
    assert captured["temperature"] == 0.0
    assert captured["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        "[]",
        _json_response(mode="unsupported"),
        _json_response(media_type="podcast"),
        _json_response(year_from=2030, year_to=2000),
        _json_response(runtime_min=200, runtime_max=90),
        _json_response(genre_terms="comedy"),
    ],
)
def test_invalid_json_or_fields_raise_intent_error(monkeypatch, response):
    _mock_classifier(monkeypatch, response)

    with pytest.raises(RecommendationIntentError):
        intent.extract_recommendation_intent("recommend something")


def test_fenced_json_is_accepted(monkeypatch):
    _mock_classifier(monkeypatch, f"```json\n{_json_response(mode='discovery')}\n```")

    result = intent.extract_recommendation_intent("something good")

    assert result.mode == "discovery"


def test_empty_input_raises_without_calling_model(monkeypatch):
    fake = pytest.fail
    monkeypatch.setattr(intent, "chat_completion", fake)

    with pytest.raises(RecommendationIntentError):
        intent.extract_recommendation_intent("   ")


def test_openrouter_failure_raises_intent_error(monkeypatch):
    def failing_chat_completion(*args, **kwargs):
        raise intent.OpenRouterError("provider unavailable")

    monkeypatch.setattr(intent, "chat_completion", failing_chat_completion)

    with pytest.raises(RecommendationIntentError, match="Intent extraction failed"):
        intent.extract_recommendation_intent("recommend a comedy")