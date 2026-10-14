import json
from typing import Any

from pydantic import ValidationError

from app.recommend.models import RecommendationIntent
from app.services.openrouter_client import OpenRouterError, chat_completion

INTENT_SYSTEM_PROMPT = """
Extract structured movie/TV recommendation intent from the user's message.
Return exactly one JSON object with these keys:
mode, media_type, reference_title, reference_year, genre_terms, mood_terms,
year_from, year_to, runtime_min, runtime_max, watch_provider_terms, language,
watch_region.

Map user intent to the three supported mode values:
- A request based on a liked/watched title ("last_watched") maps to
    mode="similarity". Set reference_title only to a title the user named or
    history explicitly identifies; otherwise use null. Do not invent titles.
- A genre, mood, or type description ("genre_mood") maps to
    mode="discovery", with explicitly requested genres in genre_terms and
    atmosphere/theme concepts in mood_terms.
- A refinement such as "funnier", "shorter", or "older" maps to
    mode="discovery" only when non-empty conversation history establishes what
    is being refined. Inherit relevant genre/mood filters from that history,
    apply the new constraint when representable, and do not set reference_title
    unless the user re-mentions it.
- A vague request such as "something good" with no genre, mood, title, or
    filters maps to mode="discovery" with empty terms and null filters.
- An unrelated request maps to mode="off_topic".

Do not infer a refinement from the current message alone when history is absent
or empty. media_type must be the string "movie", "tv", or "both"; use
"both" when unspecified, never null. Extract a clean reference year only when
stated or clearly established. Extract only filters supported by the user's
message or relevant history. Use ISO 639-1 for language and ISO 3166-1 alpha-2
for watch_region. Use null for absent scalar values and [] for absent lists.
Do not recommend titles and do not return TMDB IDs.
""".strip()


class RecommendationIntentError(RuntimeError):
    pass


def _decode_json(response: str) -> dict[str, Any]:
    content = response.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3].strip()
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RecommendationIntentError("The intent model returned invalid JSON.") from exc
    if not isinstance(value, dict):
        raise RecommendationIntentError("The intent model returned an invalid object.")
    return value


def _history_summary(history: list[dict[str, str]] | None) -> str | None:
    if not history:
        return None

    turns = []
    for turn in history[-6:]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role in {"user", "assistant"} and content.strip():
            turns.append(f"{role}: {content.strip()}")
    if not turns:
        return None
    return "Recent conversation history, oldest to newest:\n" + "\n".join(turns)


def extract_recommendation_intent(
    text: str, history: list[dict[str, str]] | None = None
) -> RecommendationIntent:
    if not text.strip():
        raise RecommendationIntentError("A recommendation request is required.")
    try:
        messages = [{"role": "system", "content": INTENT_SYSTEM_PROMPT}]
        summary = _history_summary(history)
        if summary is not None:
            messages.append({"role": "system", "content": summary})
        messages.append({"role": "user", "content": text})
        response = chat_completion(
            messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return RecommendationIntent.model_validate(_decode_json(response))
    except OpenRouterError as exc:
        raise RecommendationIntentError(f"Intent extraction failed: {exc}") from exc
    except ValidationError as exc:
        raise RecommendationIntentError("The intent model returned invalid fields.") from exc