import asyncio
from difflib import SequenceMatcher
from typing import Any

from app.recommend.enrich import WatchMemo, rank_with_optional_omdb, to_candidate
from app.recommend.models import Candidate, MediaType, ScoredCandidate
from app.recommend.tmdb_client import (
    get_details,
    get_recommendations,
    get_similar,
    search_title,
)


class TitleNotFoundError(LookupError):
    pass


def _normalized_title(value: str) -> str:
    return " ".join(value.casefold().split())


def select_title_match(
    results: list[dict[str, Any]], title: str, year: int | None = None
) -> dict[str, Any] | None:
    normalized = _normalized_title(title)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for result in results:
        result_title = result.get("title") or result.get("name")
        if not isinstance(result_title, str):
            continue
        score = SequenceMatcher(None, normalized, _normalized_title(result_title)).ratio()
        if year is not None:
            date = result.get("release_date") or result.get("first_air_date") or ""
            try:
                result_year = int(date[:4])
            except (TypeError, ValueError):
                result_year = None
            if result_year == year:
                score += 0.15
            elif result_year is not None:
                score -= 0.1
        candidates.append((score, result))
    if not candidates:
        return None
    score, result = max(candidates, key=lambda item: item[0])
    return result if score >= 0.45 else None


async def resolve_title(
    client: Any,
    media_type: MediaType,
    title: str,
    year: int | None = None,
) -> dict[str, Any] | None:
    results = await search_title(client, media_type, title, year)
    return select_title_match(results, title, year)


async def recommend_from_watched(
    client: Any,
    media_type: MediaType,
    tmdb_id: int,
    limit: int = 5,
    watch_region: str | None = None,
) -> list[ScoredCandidate]:
    source_raw = await get_details(client, media_type, tmdb_id)
    watch_memo: WatchMemo = {}
    source = await to_candidate(
        client, media_type, source_raw, watch_region=watch_region, watch_memo=watch_memo
    )
    similar, recommendations = await asyncio.gather(
        get_similar(client, media_type, tmdb_id),
        get_recommendations(client, media_type, tmdb_id),
    )

    combined = {
        result["id"]: result
        for result in (*similar, *recommendations)
        if isinstance(result.get("id"), int) and result["id"] != tmdb_id
    }
    pool: list[Candidate] = list(
        await asyncio.gather(
            *(
                to_candidate(
                    client, media_type, result, watch_region=watch_region,
                    watch_memo=watch_memo,
                )
                for result in list(combined.values())[:40]
            )
        )
    )
    return await rank_with_optional_omdb(pool, source=source, limit=limit)