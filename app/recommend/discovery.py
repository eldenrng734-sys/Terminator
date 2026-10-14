import asyncio
from typing import Any

from app.recommend.discover_params import MediaType, build_discover_params
from app.recommend.enrich import WatchMemo, rank_with_optional_omdb, to_candidate
from app.recommend.models import Candidate, ScoredCandidate
from app.recommend.scoring import MIN_AUDIENCE_SCORE, MIN_VOTE_COUNT
from app.recommend.tmdb_client import discover


async def recommend_from_query(
    client: Any,
    media_type: MediaType,
    *,
    filters: dict[str, Any],
    limit: int = 5,
) -> list[ScoredCandidate]:
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10.")

    request_filters = dict(filters)
    request_filters.setdefault("min_vote_average", MIN_AUDIENCE_SCORE)
    request_filters.setdefault("min_vote_count", MIN_VOTE_COUNT)
    watch_region = request_filters.get("watch_region")
    watch_memo: WatchMemo = {}

    async def search(current: dict[str, Any]) -> list[ScoredCandidate]:
        params = build_discover_params(media_type, **current)
        raw_results = await discover(client, media_type, **params)
        pool: list[Candidate] = list(
            await asyncio.gather(
                *(
                    to_candidate(
                        client, media_type, result, watch_region=watch_region,
                        watch_memo=watch_memo,
                    )
                    for result in raw_results[:40]
                )
            )
        )
        return await rank_with_optional_omdb(pool, source=None, limit=limit)

    results = await search(request_filters)
    if results:
        return results

    keyword_filters = dict(request_filters)
    had_keywords = bool(keyword_filters.pop("keyword_ids_any", None))
    if had_keywords:
        results = await search(keyword_filters)
        if results:
            return results

    relaxed_filters = dict(keyword_filters if had_keywords else request_filters)
    current_vote_floor = int(relaxed_filters.get("min_vote_count", MIN_VOTE_COUNT))
    if current_vote_floor > 50:
        relaxed_filters["min_vote_count"] = 50
        return await search(relaxed_filters)
    return []