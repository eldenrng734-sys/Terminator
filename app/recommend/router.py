import asyncio
import inspect
import json
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import require_user
from app.config import get_settings
from app.recommend.discovery import recommend_from_query
from app.recommend.intent import RecommendationIntentError, extract_recommendation_intent
from app.recommend.keywords import (
    resolve_genre_ids,
    resolve_keyword_ids,
    resolve_watch_provider_ids,
)
from app.recommend.models import (
    RecommendationIntent,
    RecommendationRequest,
    RecommendationResponse,
    ScoredCandidate,
)
from app.recommend.scoring import MIN_VOTE_COUNT
from app.recommend.similarity import recommend_from_watched, resolve_title
from app.recommend.tmdb_client import TMDBError, create_client
from app.services.llm import SYSTEM_PROMPT
from app.services.openrouter_client import OpenRouterError, chat_completion
from app.services.session import session_store

router = APIRouter(prefix="/api", tags=["recommendations"])
_REFINEMENT_CUE = re.compile(
    r"\b(?:funnier|shorter|longer|older|newer|darker|lighter|scarier|"
    r"happier|sadder|more|less|another|different|same)\b",
    re.IGNORECASE,
)


def build_candidate_context(
    candidates: list[ScoredCandidate],
) -> list[dict[str, Any]]:
    """Minimal verified facts drawn only from actual candidate objects."""
    context: list[dict[str, Any]] = []
    for candidate in candidates:
        entry: dict[str, Any] = {
            "tmdb_id": candidate.tmdb_id,
            "media_type": candidate.media_type,
            "title": candidate.title,
        }
        if candidate.year is not None:
            entry["year"] = candidate.year
        if candidate.vote_count and candidate.vote_average:
            entry["vote_average"] = candidate.vote_average
            entry["vote_count"] = candidate.vote_count
        if candidate.imdb_rating is not None:
            entry["imdb_rating"] = candidate.imdb_rating
        if candidate.rt_score is not None:
            entry["rt_score"] = candidate.rt_score
        if candidate.metacritic_score is not None:
            entry["metacritic_score"] = candidate.metacritic_score
        if candidate.runtime is not None:
            entry["runtime"] = candidate.runtime
        if candidate.original_language:
            entry["original_language"] = candidate.original_language
        if candidate.overview:
            entry["overview"] = candidate.overview[:500]
        if candidate.reasons:
            entry["reasons"] = list(candidate.reasons)
        if candidate.watch_providers:
            entry["watch_providers"] = list(candidate.watch_providers)
        if candidate.watch_providers_by_category:
            entry["watch_providers_by_category"] = {
                category: list(names)
                for category, names in candidate.watch_providers_by_category.items()
            }
        if candidate.watch_region:
            entry["watch_region"] = candidate.watch_region
        if candidate.watch_link:
            entry["watch_link"] = candidate.watch_link
        context.append(entry)
    return context


def _fallback_reply(candidates: list[ScoredCandidate]) -> str:
    return "Here are the strongest matches I found: " + "; ".join(
        f"{candidate.title} ({candidate.year or 'year unavailable'})"
        for candidate in candidates
    )


def _decode_selections(raw: str) -> list[dict[str, Any]] | None:
    content = raw.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3].strip()
    try:
        value: Any = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(value, dict):
        value = value.get("selections")
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, dict)] or []


def _render_verified_reply(
    candidates: list[ScoredCandidate],
    selections: list[dict[str, Any]],
) -> str | None:
    lookup = {(c.media_type, c.tmdb_id): c for c in candidates}
    verified: list[tuple[ScoredCandidate, str]] = []
    for item in selections:
        tmdb_id = item.get("tmdb_id", item.get("id"))
        media_type = item.get("media_type")
        if isinstance(tmdb_id, bool) or not isinstance(tmdb_id, int):
            continue
        if media_type not in ("movie", "tv"):
            continue
        candidate = lookup.get((media_type, tmdb_id))
        if candidate is None:
            continue
        verified.append((candidate, ""))
    if not verified:
        return None
    blocks: list[str] = []
    for index, (candidate, why) in enumerate(verified, start=1):
        year = f" ({candidate.year})" if candidate.year else ""
        blocks.append(f"### {index}. {candidate.title}{year}")
        if candidate.reasons:
            blocks.append("Why it fits: " + ", ".join(candidate.reasons[:3]) + ".")
        else:
            blocks.append(
                "Why it fits: It meets the quality filters with a "
                f"TMDB rating of {candidate.vote_average:.1f} from "
                f"{candidate.vote_count} votes."
            )
        facts: list[str] = [candidate.media_type]
        if candidate.vote_count and candidate.vote_average:
            facts.append(
                f"rating {candidate.vote_average:.1f} ({candidate.vote_count} votes)"
            )
        if candidate.runtime is not None:
            facts.append(f"{candidate.runtime} min")
        if candidate.reasons:
            facts.append("; ".join(candidate.reasons[:3]))
        blocks.append("_" + " · ".join(facts) + "_")
    return "\n\n".join(blocks)


async def _call_write_reply(
    query: str,
    candidates: list[ScoredCandidate],
    history: list[dict[str, Any]] | None,
) -> str:
    """Call _write_reply tolerating legacy 2-arg test doubles."""
    try:
        params = inspect.signature(_write_reply).parameters.values()
        accepts_history = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            or p.kind == inspect.Parameter.VAR_POSITIONAL
            or p.name == "history"
            for p in params
        )
    except (TypeError, ValueError):
        accepts_history = True
    if accepts_history:
        return await _write_reply(query, candidates, history=history)
    return await _write_reply(_query_with_history(query, history), candidates)


# Section 3 missing; using SYSTEM_PROMPT as replacement (no docs/*.md in repo).
_FINAL_REPLY_STYLE = (
    "FINAL REPLY: explain why each pick fits the request, respect refinements, "
    "use readable Markdown, keep it focused (2-4 sentences per pick, 2-3 picks). "
    "Select only from the CANDIDATE LIST."
)
_SELECTION_INSTRUCTIONS = (
    'Reply ONLY with one JSON object: {"selections": '
    '[{"tmdb_id": <int>, "media_type": "<movie|tv>"}]}. '
    "Select only titles from the CANDIDATE LIST. Do not include titles, "
    "explanations, ratings, or metadata in the selection object."
)
_EMPTY_SHORTLIST_REPLY = (
    "I couldn't find a strong match in the available titles. "
    "Try a broader genre or mood."
)


async def _write_reply(
    query: str,
    candidates: list[ScoredCandidate],
    history: list[dict[str, Any]] | None = None,
) -> str:
    if not candidates:
        return _EMPTY_SHORTLIST_REPLY

    candidate_data = json.dumps(
        build_candidate_context(candidates),
        ensure_ascii=False,
    )
    user_content = _query_with_history(query, history) if history else query
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{_FINAL_REPLY_STYLE}"},
        {
            "role": "system",
            "content": (
                "CANDIDATE LIST (recommend only these titles):\n"
                f"{candidate_data}\n{_SELECTION_INSTRUCTIONS}"
            ),
        },
        {"role": "user", "content": user_content},
    ]
    try:
        raw = await asyncio.to_thread(
            chat_completion,
            messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except OpenRouterError:
        return _fallback_reply(candidates)
    selections = _decode_selections(raw)
    if selections is None:
        return _fallback_reply(candidates)
    rendered = _render_verified_reply(candidates, selections)
    if rendered is None:
        return _fallback_reply(candidates)
    return rendered


def _media_types(requested: str) -> list[str]:
    return ["movie", "tv"] if requested == "both" else [requested]


def _merge_results(
    results: list[list[ScoredCandidate]], limit: int
) -> list[ScoredCandidate]:
    unique: dict[tuple[str, int], ScoredCandidate] = {}
    for group in results:
        for candidate in group:
            unique.setdefault((candidate.media_type, candidate.tmdb_id), candidate)
    ordered = sorted(
        unique.values(),
        key=lambda item: (-item.score, -item.vote_count),
    )
    return ordered[:limit]


def _is_refinement(query: str) -> bool:
    return _REFINEMENT_CUE.search(query) is not None


def _previous_user_request(
    history: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]] | None:
    for index in range(len(history) - 1, -1, -1):
        turn = history[index]
        content = turn.get("content")
        if turn.get("role") == "user" and isinstance(content, str) and content.strip():
            return content, history[:index]
    return None


def _intent_history(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, str]] | None:
    if not history:
        return None

    text_history: list[dict[str, str]] = []
    for turn in history[-12:]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            )
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            text_history.append({"role": role, "content": content.strip()})
    return text_history or None


def _merge_refinement_filters(
    current: RecommendationIntent,
    previous: RecommendationIntent,
) -> RecommendationIntent:
    def merge_terms(old_terms: list[str], new_terms: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for term in (*old_terms, *new_terms):
            normalized = term.casefold().strip()
            if normalized and normalized not in seen:
                merged.append(term)
                seen.add(normalized)
        return merged

    current_data = current.model_dump()
    previous_data = previous.model_dump()
    updates: dict[str, Any] = {
        "genre_terms": merge_terms(previous.genre_terms, current.genre_terms),
        "mood_terms": merge_terms(previous.mood_terms, current.mood_terms),
        "watch_provider_terms": merge_terms(
            previous.watch_provider_terms, current.watch_provider_terms
        ),
    }
    if current.media_type == "both":
        updates["media_type"] = previous.media_type

    for first, second in (("year_from", "year_to"), ("runtime_min", "runtime_max")):
        first_value = current_data[first]
        second_value = current_data[second]
        if first_value is None and second_value is None:
            first_value = previous_data[first]
            second_value = previous_data[second]
        else:
            first_value = first_value if first_value is not None else previous_data[first]
            second_value = second_value if second_value is not None else previous_data[second]
            if (
                first_value is not None
                and second_value is not None
                and first_value > second_value
            ):
                first_value = current_data[first]
                second_value = current_data[second]
        updates[first] = first_value
        updates[second] = second_value
    for field in ("language", "watch_region"):
        if current_data[field] is None:
            updates[field] = previous_data[field]

    return RecommendationIntent.model_validate({**current_data, **updates})


def _query_with_history(
    query: str, history: list[dict[str, Any]] | None
) -> str:
    current = f"Current user request: {query.strip()}"[:1000]
    if not history:
        return current

    previous = [
        f"{turn.get('role', 'user')}: {content.strip()}"
        for turn in history[-6:]
        if isinstance((content := turn.get("content")), str) and content.strip()
    ]
    budget = max(0, 1000 - len(current) - 1)
    context = "\n".join(previous)[-budget:] if budget else ""
    return f"{context}\n{current}" if context else current


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(
    payload: RecommendationRequest,
    session_id: str | None = None,
    surprise: bool = Query(False),
    user_id: int = Depends(require_user),
) -> RecommendationResponse:
    if session_id and session_store.get_session(session_id, user_id=user_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    history = (
        session_store.load_recent_turns(session_id, user_id=user_id, limit=12)
        if session_id
        else None
    )
    result = await run_recommendation(payload, history=history, surprise=surprise)
    if session_id:
        session_store.append(
            session_id,
            {"role": "user", "content": payload.query},
            result.reply,
            user_id=user_id,
        )
    return result


async def _run_surprise_recommendation(
    payload: RecommendationRequest,
    history: list[dict[str, Any]] | None,
) -> RecommendationResponse:
    settings = get_settings()
    watch_region = (payload.watch_region or settings.default_watch_region)
    if watch_region:
        watch_region = watch_region.upper()
    filters: dict[str, Any] = {
        "genre_ids": [],
        "keyword_ids_any": [],
        "watch_provider_ids": [],
        "year_from": None,
        "year_to": None,
        "runtime_min": None,
        "runtime_max": None,
        "language": None,
        "watch_region": watch_region,
        "min_vote_average": settings.surprise_min_vote_average,
        "min_vote_count": MIN_VOTE_COUNT,
    }
    media_types = _media_types(payload.media_type or "both")
    try:
        async with create_client() as client:
            result_groups = [
                await recommend_from_query(
                    client, media_type, filters=filters, limit=payload.limit
                )
                for media_type in media_types
            ]
    except TMDBError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    candidates = _merge_results(result_groups, payload.limit)
    return RecommendationResponse(
        mode="discovery",
        reply=await _call_write_reply(payload.query, candidates, history),
        candidates=candidates,
    )


async def run_recommendation(
    payload: RecommendationRequest,
    history: list[dict[str, Any]] | None = None,
    surprise: bool = False,
) -> RecommendationResponse:
    if surprise:
        return await _run_surprise_recommendation(payload, history)

    intent_history = _intent_history(history)
    try:
        intent = await asyncio.to_thread(
            extract_recommendation_intent,
            payload.query,
            history=intent_history,
        )
    except RecommendationIntentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if intent.mode == "off_topic":
        raise HTTPException(
            status_code=422,
            detail="This endpoint handles movie and TV recommendations.",
        )

    if intent.mode == "similarity" and not intent.reference_title:
        raise HTTPException(
            status_code=422,
            detail="I couldn't identify the title you want recommendations from.",
        )

    if intent_history and _is_refinement(payload.query):
        previous_request = _previous_user_request(intent_history)
        if previous_request is not None:
            previous_text, earlier_history = previous_request
            try:
                previous_intent = await asyncio.to_thread(
                    extract_recommendation_intent,
                    previous_text,
                    history=earlier_history or None,
                )
            except RecommendationIntentError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            if previous_intent is not None and previous_intent.mode == "discovery":
                intent = _merge_refinement_filters(intent, previous_intent)

    media_types = _media_types(payload.media_type or intent.media_type)
    watch_region = (
        payload.watch_region or intent.watch_region or get_settings().default_watch_region
    )
    if watch_region:
        watch_region = watch_region.upper()

    try:
        async with create_client() as client:
            if intent.mode == "similarity":
                result_groups: list[list[ScoredCandidate]] = []
                for media_type in media_types:
                    match = await resolve_title(
                        client,
                        media_type,
                        intent.reference_title,
                        intent.reference_year,
                    )
                    if match is not None:
                        result_groups.append(
                            await recommend_from_watched(
                                client,
                                media_type,
                                match["id"],
                                limit=payload.limit,
                                watch_region=watch_region,
                            )
                        )
                candidates = _merge_results(result_groups, payload.limit)
                mode = "similarity"
            else:
                result_groups = []
                for media_type in media_types:
                    genre_ids = await resolve_genre_ids(
                        client, media_type, intent.genre_terms
                    )
                    keyword_ids = await resolve_keyword_ids(
                        client, intent.mood_terms
                    )
                    watch_provider_ids = await resolve_watch_provider_ids(
                        client,
                        media_type,
                        intent.watch_provider_terms,
                        watch_region,
                    )
                    filters: dict[str, Any] = {
                        "genre_ids": genre_ids,
                        "keyword_ids_any": keyword_ids,
                        "watch_provider_ids": watch_provider_ids,
                        "year_from": intent.year_from,
                        "year_to": intent.year_to,
                        "runtime_min": intent.runtime_min,
                        "runtime_max": intent.runtime_max,
                        "language": intent.language,
                        "watch_region": watch_region,
                    }
                    result_groups.append(
                        await recommend_from_query(
                            client,
                            media_type,
                            filters=filters,
                            limit=payload.limit,
                        )
                    )
                candidates = _merge_results(result_groups, payload.limit)
                mode = "discovery"
    except TMDBError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return RecommendationResponse(
        mode=mode,
        reply=await _call_write_reply(payload.query, candidates, history),
        candidates=candidates,
    )