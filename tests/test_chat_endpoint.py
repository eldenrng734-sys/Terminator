from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from app.auth import require_user
from app.services.session import SessionStore


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(
        main, "session_store", SessionStore(db_path=tmp_path / "sessions.sqlite3")
    )
    monkeypatch.setitem(main.app.dependency_overrides, require_user, lambda: 1)
    return TestClient(main.app)


def test_chat_unknown_session_returns_404(client: TestClient) -> None:
    response = client.post(
        "/api/chat", data={"message": "Hi", "session_id": "missing-session"}
    )

    assert response.status_code == 404
    assert "Conversation not found" in response.json()["detail"]


def test_chat_off_topic_falls_back_to_general_reply(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    monkeypatch.setattr(
        main, "build_user_message", lambda text, images: {"role": "user", "content": text}
    )

    async def off_topic(request, history=None, surprise=False):
        raise HTTPException(status_code=422, detail="off-topic")

    monkeypatch.setattr(main, "run_recommendation", off_topic)
    monkeypatch.setattr(main, "get_chat_reply", lambda messages: "General hello.")

    response = client.post("/api/chat", data={"message": "weather today?"})

    assert response.status_code == 200
    assert response.json()["reply"] == "General hello."


def test_chat_provider_failure_leaks_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main, "build_user_message", lambda text, images: {"role": "user", "content": text}
    )

    async def failing_recommendation(request, history=None, surprise=False):
        raise main.HTTPException(status_code=502, detail="TMDB unavailable")

    monkeypatch.setattr(main, "run_recommendation", failing_recommendation)

    response = client.post("/api/chat", data={"message": "A tense thriller"})

    assert response.status_code == 502
    body = response.text
    assert "sk-" not in body and "Bearer" not in body and "Traceback" not in body


def test_chat_success_shape_matches_frontend(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main, "build_user_message", lambda text, images: {"role": "user", "content": text}
    )

    async def fake_recommendation(request, history=None, surprise=False):
        return SimpleNamespace(reply="Hello from Gemma.")

    monkeypatch.setattr(main, "run_recommendation", fake_recommendation)

    response = client.post("/api/chat", data={"message": "Hi"})

    assert response.status_code == 200
    assert set(response.json()) == {"session_id", "reply"}


def test_chat_bare_surprise_me_uses_surprise_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main, "build_user_message", lambda text, images: {"role": "user", "content": text}
    )
    seen = {}

    async def fake_recommendation(request, history=None, surprise=False):
        seen["surprise"] = surprise
        return SimpleNamespace(reply="Surprise pick.")

    monkeypatch.setattr(main, "run_recommendation", fake_recommendation)

    for message in ("surprise me", "Surprise me!", "please surprise me"):
        response = client.post("/api/chat", data={"message": message})

        assert response.status_code == 200
        assert response.json()["reply"] == "Surprise pick."
        assert seen["surprise"] is True


def test_chat_surprise_with_genre_keeps_intent_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main, "build_user_message", lambda text, images: {"role": "user", "content": text}
    )
    seen = {}

    async def fake_recommendation(request, history=None, surprise=False):
        seen["surprise"] = surprise
        return SimpleNamespace(reply="Horror pick.")

    monkeypatch.setattr(main, "run_recommendation", fake_recommendation)

    response = client.post("/api/chat", data={"message": "surprise me with a comedy"})

    assert response.status_code == 200
    assert seen["surprise"] is False
