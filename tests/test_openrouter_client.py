from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from app.services import openrouter_client


@pytest.fixture
def mocked_openrouter(monkeypatch):
    settings = SimpleNamespace(
        openrouter_api_key="test-openrouter-key",
        llm_model="configured-model",
    )
    monkeypatch.setattr(openrouter_client, "get_settings", lambda: settings)

    client = MagicMock()
    context_manager = MagicMock()
    context_manager.__enter__.return_value = client
    context_manager.__exit__.return_value = False
    client_factory = MagicMock(return_value=context_manager)
    monkeypatch.setattr(openrouter_client.httpx, "Client", client_factory)
    return settings, client, client_factory


def _response(status_code=200, *, json_body=None, content=None, text=None):
    request = httpx.Request(
        "POST", f"{openrouter_client.OPENROUTER_BASE_URL}/chat/completions"
    )
    if text is not None:
        return httpx.Response(status_code, text=text, request=request)
    if content is not None:
        return httpx.Response(status_code, content=content, request=request)
    return httpx.Response(status_code, json=json_body, request=request)


def test_success_uses_configured_model_and_default_payload(mocked_openrouter):
    _, client, client_factory = mocked_openrouter
    client.post.return_value = _response(
        json_body={"choices": [{"message": {"content": "A useful answer."}}]}
    )

    result = openrouter_client.chat_completion([{"role": "user", "content": "Hi"}])

    assert result == "A useful answer."
    client_factory.assert_called_once_with(
        base_url=openrouter_client.OPENROUTER_BASE_URL,
        headers={
            "Authorization": "Bearer test-openrouter-key",
            "Content-Type": "application/json",
        },
        timeout=60.0,
    )
    client.post.assert_called_once_with(
        "/chat/completions",
        json={
            "model": "configured-model",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )


def test_model_override_is_forwarded(mocked_openrouter):
    _, client, _ = mocked_openrouter
    client.post.return_value = _response(
        json_body={"choices": [{"message": {"content": "ok"}}]}
    )

    openrouter_client.chat_completion([], "override-model")

    assert client.post.call_args.kwargs["json"]["model"] == "override-model"


def test_optional_temperature_and_response_format_are_forwarded(mocked_openrouter):
    _, client, _ = mocked_openrouter
    client.post.return_value = _response(
        json_body={"choices": [{"message": {"content": "{}"}}]}
    )

    openrouter_client.chat_completion(
        [], temperature=0.0, response_format={"type": "json_object"}
    )

    assert client.post.call_args.kwargs["json"]["temperature"] == 0.0
    assert client.post.call_args.kwargs["json"]["response_format"] == {
        "type": "json_object"
    }


def test_optional_fields_are_omitted_when_unset(mocked_openrouter):
    _, client, _ = mocked_openrouter
    client.post.return_value = _response(
        json_body={"choices": [{"message": {"content": "ok"}}]}
    )

    openrouter_client.chat_completion([])

    payload = client.post.call_args.kwargs["json"]
    assert "temperature" not in payload
    assert "response_format" not in payload


def test_http_error_response_raises_with_truncated_redacted_body(mocked_openrouter):
    _, client, _ = mocked_openrouter
    client.post.return_value = _response(
        429,
        text="test-openrouter-key " + ("x" * 600),
    )

    with pytest.raises(openrouter_client.OpenRouterError) as error:
        openrouter_client.chat_completion([])

    assert "429" in str(error.value)
    assert "test-openrouter-key" not in str(error.value)
    assert len(str(error.value).split(": ", 1)[1]) <= 500


@pytest.mark.parametrize(
    "response",
    [
        _response(content=b"not-json"),
        _response(json_body={"choices": []}),
        _response(json_body={"choices": [{"message": {"content": ""}}]}),
        _response(json_body={"choices": [{"message": {"content": "  "}}]}),
        _response(json_body={"choices": [{"message": {"content": 3}}]}),
    ],
)
def test_malformed_or_empty_content_raises(mocked_openrouter, response):
    _, client, _ = mocked_openrouter
    client.post.return_value = response

    with pytest.raises(openrouter_client.OpenRouterError, match="Unexpected OpenRouter response"):
        openrouter_client.chat_completion([])


def test_httpx_transport_error_raises_openrouter_error(mocked_openrouter):
    _, client, _ = mocked_openrouter
    client.post.side_effect = httpx.ConnectError("connection failed")

    with pytest.raises(openrouter_client.OpenRouterError, match="request failed"):
        openrouter_client.chat_completion([])


def test_post_is_mocked_and_no_network_request_is_made(mocked_openrouter):
    _, client, _ = mocked_openrouter
    client.post.return_value = _response(
        json_body={"choices": [{"message": {"content": "mocked"}}]}
    )

    openrouter_client.chat_completion([])

    client.post.assert_called_once()