import asyncio
import re
from collections.abc import Iterable
from typing import Any

from app.recommend.models import MediaType
from app.recommend.tmdb_client import (
    get_genres,
    get_watch_provider_list,
    search_keyword,
)

GENRE_ALIASES = {
    "action comedy": "action",
    "rom com": "romance",
    "romcom": "romance",
    "sci fi": "science fiction",
    "sci-fi": "science fiction",
    "psychological thriller": "thriller",
    "true crime": "crime",
}

MOOD_SEARCH_TERMS = {
    "heartbreaking": ("tragedy", "grief", "loss"),
    "heartbroken": ("tragedy", "grief", "loss"),
    "cozy": ("feel good", "friendship", "small town"),
    "comforting": ("feel good", "friendship", "family"),
    "tense": ("suspense", "paranoia", "thriller"),
    "suspenseful": ("suspense", "paranoia", "thriller"),
    "heist": ("heist", "robbery"),
    "mind bending": ("mind-bending", "time loop", "reality"),
    "mind-bending": ("mind-bending", "time loop", "reality"),
    "uplifting": ("feel good", "inspirational", "overcoming adversity"),
    "romantic": ("romance", "love", "relationship"),
}

WATCH_PROVIDER_ALIASES = {
    "amazon": "amazon prime video",
    "prime": "amazon prime video",
    "prime video": "amazon prime video",
    "disney+": "disney plus",
    "hbo": "max",
}


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _nonempty_terms(terms: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not isinstance(term, str):
            continue
        cleaned = term.strip()
        normalized = _normalize(cleaned)
        if normalized and normalized not in seen:
            result.append(cleaned)
            seen.add(normalized)
    return result


async def resolve_genre_ids(
    client: Any, media_type: MediaType, genre_terms: Iterable[str]
) -> list[int]:
    genres = await get_genres(client, media_type)
    genre_lookup = {_normalize(item["name"]): item["id"] for item in genres}
    genre_ids: list[int] = []
    for term in _nonempty_terms(genre_terms):
        normalized = _normalize(term)
        normalized = _normalize(GENRE_ALIASES.get(normalized, normalized))
        genre_id = genre_lookup.get(normalized)
        if genre_id is not None and genre_id not in genre_ids:
            genre_ids.append(genre_id)
    return genre_ids


def _queries_for_moods(mood_terms: Iterable[str]) -> list[str]:
    queries: list[str] = []
    for term in _nonempty_terms(mood_terms):
        normalized = _normalize(term)
        queries.append(term)
        queries.extend(MOOD_SEARCH_TERMS.get(normalized, ()))
    return _nonempty_terms(queries)[:8]


async def resolve_keyword_ids(
    client: Any, mood_terms: Iterable[str]
) -> list[int]:
    queries = _queries_for_moods(mood_terms)
    if not queries:
        return []

    results = await asyncio.gather(*(search_keyword(client, query) for query in queries))
    keyword_ids: list[int] = []
    for query, matches in zip(queries, results):
        if not matches:
            continue
        normalized_query = _normalize(query)
        exact = next(
            (
                match
                for match in matches
                if _normalize(str(match.get("name", ""))) == normalized_query
                and isinstance(match.get("id"), int)
            ),
            None,
        )
        selected = exact or next(
            (match for match in matches if isinstance(match.get("id"), int)), None
        )
        if selected is not None and selected["id"] not in keyword_ids:
            keyword_ids.append(selected["id"])
    return keyword_ids


async def resolve_watch_provider_ids(
    client: Any,
    media_type: MediaType,
    provider_terms: Iterable[str],
    watch_region: str | None,
) -> list[int]:
    terms = _nonempty_terms(provider_terms)
    if not terms or not watch_region:
        return []

    providers = await get_watch_provider_list(client, media_type, watch_region)
    provider_ids: list[int] = []
    for term in terms:
        normalized = _normalize(term)
        normalized = _normalize(WATCH_PROVIDER_ALIASES.get(normalized, normalized))
        matches = [
            provider
            for provider in providers
            if isinstance(provider.get("provider_id"), int)
            and normalized
            in _normalize(str(provider.get("provider_name", "")))
        ]
        for provider in matches[:2]:
            provider_id = provider["provider_id"]
            if provider_id not in provider_ids:
                provider_ids.append(provider_id)
    return provider_ids