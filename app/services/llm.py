import base64
from typing import Any

from app.config import get_settings
from app.services.openrouter_client import OpenRouterError, chat_completion


class LLMError(RuntimeError):
    pass


SYSTEM_PROMPT = """
You are a professional AI assistant. Respond clearly and naturally using Markdown.

- Use **bold** for important terms, figures, and conclusions.
- Use ## and ### headings for longer answers.
- Use bullet points for lists and numbered steps for instructions.
- Use tables for comparisons when helpful.
- Keep answers concise, organized, and easy to scan.
- For images and dashboards, focus on key findings and insights, not just visual descriptions.
- Avoid repetition and unnecessary introductions.
- Never invent facts or alter figures.
- Return Markdown, never wrap the entire response in a code block.
""".strip()


def build_user_message(
    text: str,
    images: list[tuple[bytes, str]] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    images = images or []
    if not text.strip() and not images:
        raise LLMError("Message needs text or at least one image.")

    if not images:
        return {"role": "user", "content": text}

    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    max_bytes = settings.max_image_mb * 1024 * 1024
    for data, mime in images:
        if mime not in settings.allowed_image_types:
            raise LLMError(f"Unsupported image type: {mime}")
        if len(data) > max_bytes:
            raise LLMError(f"Image exceeds {settings.max_image_mb} MB limit.")
        b64 = base64.b64encode(data).decode("ascii")
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )
    return {"role": "user", "content": parts}


def get_chat_reply(messages: list[dict[str, Any]]) -> str:
    if not messages:
        raise LLMError("No conversation history provided.")
    try:
        return chat_completion(
            [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        )
    except OpenRouterError as exc:
        raise LLMError(str(exc)) from exc
