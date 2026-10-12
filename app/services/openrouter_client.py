from functools import lru_cache
from typing import Any

import httpx

from app.config import get_settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    pass


def get_openrouter_headers() -> dict[str, str]:
    settings = get_settings()
    return {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }


@lru_cache
def get_openrouter_client() -> httpx.Client:
    return httpx.Client(
        base_url=OPENROUTER_BASE_URL,
        headers=get_openrouter_headers(),
        timeout=60.0,
    )


def chat_completion(
    messages: list[dict[str, Any]],
    model: str | None = None,
    timeout: float = 60.0,
) -> str:
    settings = get_settings()
    payload = {
        "model": model or settings.llm_model,
        "messages": messages,
    }
    try:
        with httpx.Client(
            base_url=OPENROUTER_BASE_URL,
            headers=get_openrouter_headers(),
            timeout=timeout,
        ) as client:
            response = client.post("/chat/completions", json=payload)
    except httpx.HTTPError as exc:
        raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

    if response.status_code >= 400:
        raise OpenRouterError(
            f"OpenRouter error {response.status_code}: {response.text[:500]}"
        )

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError(f"Unexpected OpenRouter response: {response.text[:500]}") from exc
