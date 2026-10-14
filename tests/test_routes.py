from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

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


def test_api_routes_require_authentication() -> None:
    with TestClient(main.app) as unauthenticated_client:
        responses = [
            unauthenticated_client.get("/api/profile"),
            unauthenticated_client.get("/api/sessions"),
            unauthenticated_client.post("/api/sessions"),
            unauthenticated_client.get("/api/sessions/missing"),
            unauthenticated_client.delete("/api/sessions/missing"),
            unauthenticated_client.post("/api/chat", data={"message": "Hi"}),
        ]

    assert [response.status_code for response in responses] == [401] * 6


def test_health_route_is_public(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_route_returns_reply(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main, "build_user_message", lambda text, images: {"role": "user", "content": text}
    )
    monkeypatch.setattr(
        main.session_store, "new_session", lambda **kwargs: "session-1"
    )
    monkeypatch.setattr(
        main.session_store, "get_history", lambda session_id, **kwargs: []
    )
    monkeypatch.setattr(main.session_store, "append", lambda *args, **kwargs: None)
    async def fake_recommendation(request, history=None, surprise=False):
        return SimpleNamespace(reply="Hello from Gemma.")

    monkeypatch.setattr(main, "run_recommendation", fake_recommendation)

    response = client.post("/api/chat", data={"message": "Hi"})

    assert response.status_code == 200
    assert response.json() == {"session_id": "session-1", "reply": "Hello from Gemma."}


def test_chat_route_uses_recommendation_pipeline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main, "build_user_message", lambda text, images: {"role": "user", "content": text}
    )
    monkeypatch.setattr(
        main.session_store, "new_session", lambda **kwargs: "session-1"
    )
    monkeypatch.setattr(
        main.session_store, "get_history", lambda session_id, **kwargs: []
    )
    monkeypatch.setattr(main.session_store, "append", lambda *args, **kwargs: None)

    async def fake_recommendation(request, history=None, surprise=False):
        assert request.query == "A tense thriller"
        assert history == []
        assert surprise is False
        return type("Recommendation", (), {"reply": "Start with this thriller."})()

    monkeypatch.setattr(main, "run_recommendation", fake_recommendation)
    monkeypatch.setattr(
        main,
        "get_chat_reply",
        lambda messages: (_ for _ in ()).throw(AssertionError("unexpected chat fallback")),
    )

    response = client.post("/api/chat", data={"message": "A tense thriller"})

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "session-1",
        "reply": "Start with this thriller.",
    }


def test_chat_route_returns_400_for_invalid_message(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main,
        "build_user_message",
        Mock(side_effect=main.LLMError("Message needs text or at least one image.")),
    )

    response = client.post("/api/chat", data={"message": ""})

    assert response.status_code == 400


def test_chat_route_returns_502_for_provider_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main, "build_user_message", lambda text, images: {"role": "user", "content": text}
    )
    monkeypatch.setattr(
        main.session_store, "new_session", lambda **kwargs: "session-1"
    )
    monkeypatch.setattr(
        main.session_store, "get_history", lambda session_id, **kwargs: []
    )
    monkeypatch.setattr(
        main,
        "get_chat_reply",
        Mock(side_effect=main.LLMError("OpenRouter unavailable")),
    )

    response = client.post(
        "/api/chat",
        data={"message": "Hi"},
        files={"images": ("test.png", b"image", "image/png")},
    )

    assert response.status_code == 502


def test_server_speech_routes_are_removed(client: TestClient) -> None:
    assert client.post("/api/stt").status_code == 404
    assert client.post("/api/tts").status_code == 404


def test_stylesheet_route_returns_css(client: TestClient) -> None:
    response = client.get("/styles.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")


def test_session_routes_create_search_and_load_history(client: TestClient) -> None:
    created = client.post("/api/sessions").json()
    search_text = f"find-{uuid4().hex}"
    main.session_store.append(
        created["id"],
        {"role": "user", "content": f"{search_text} conversation"},
        "A reply",
        user_id=1,
    )

    listed = client.get("/api/sessions", params={"q": search_text})
    loaded = client.get(f"/api/sessions/{created['id']}")

    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [created["id"]]
    assert loaded.json()["messages"] == [
        {"role": "user", "content": f"{search_text} conversation"},
        {"role": "assistant", "content": "A reply"},
    ]


def test_session_route_returns_404_for_unknown_session(client: TestClient) -> None:
    response = client.get("/api/sessions/missing")

    assert response.status_code == 404


def test_delete_session_removes_session_and_messages(client: TestClient) -> None:
    created = client.post("/api/sessions").json()
    main.session_store.append(
        created["id"],
        {"role": "user", "content": "Delete this conversation"},
        "A reply",
        user_id=1,
    )

    response = client.delete(f"/api/sessions/{created['id']}")

    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert client.get(f"/api/sessions/{created['id']}").status_code == 404
    assert all(item["id"] != created["id"] for item in client.get("/api/sessions").json())