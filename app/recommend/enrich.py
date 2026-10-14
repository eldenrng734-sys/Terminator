import asyncio
from typing import Any

import httpx

from app.config import get_settings
from app.recommend.models import Candidate, ScoredCandidate
from app.recommend.tmdb_client import TMDBError, get_details, get_watch_providers
from app.services.omdb_client import OMDbError, get_title_ratings

WATCH_CATEGORIES = ("flatrate", "free", "ads", "rent", "buy")

# Per-request memo key: (media_type, tmdb_id, region) -> (providers, by_category, link)
WatchMemo = dict[tuple[str, int, str], tuple[list[str], dict[str, list[str]], str | None]]


def _year(value: Any) -> int | None:
    if not isinstance(value, str) or len(value) < 4:
        return None
    try:
        return int(value[:4])
    except ValueError:
        return None


def _rating_value(value: str | None, *, percent: bool = False) -> int | float | None:
    if value is None:
        return None
    cleaned = value.strip().rstrip("%")
    if "/" in cleaned:
        cleaned = cleaned.split("/", 1)[0]
    try:
        return int(cleaned) if percent else float(cleaned)
    except ValueError:
        return None


def effective_watch_region(watch_region: str | None) -> str:
    candidate = (watch_region or "").strip().upper()
    if len(candidate) == 2:
        return candidate
    fallback = (get_settings().default_watch_region or "US").strip().upper()
    return fallback if len(fallback) == 2 else "US"


def _parse_region_providers(
    region_data: Any,
) -> tuple[list[str], dict[str, list[str]], str | None]:
    if not isinstance(region_data, dict):
        return [], {}, None
    link = region_data.get("link")
    if not isinstance(link, str) or not link:
        link = None
    by_category: dict[str, list[str]] = {}
    providers: list[str] = []
    seen_global: set[Any] = set()
    for category in WATCH_CATEGORIES:
        entries = region_data.get(category, [])
        if not isinstance(entries, list):
            continue
        names: list[str] = []
        seen_in_category: set[Any] = set()
        for provider in entries:
            if not isinstance(provider, dict):
                continue
            name = provider.get("provider_name")
            if not isinstance(name, str) or not name.strip():
                continue
            provider_id = provider.get("provider_id")
            key: Any = (
                provider_id
                if isinstance(provider_id, int)
                else f"name:{name.strip().casefold()}"
            )
            if key in seen_in_category:
                continue
            seen_in_category.add(key)
            names.append(name)
            if key not in seen_global:
                seen_global.add(key)
                providers.append(name)
        if names:
            by_category[category] = names
    return providers, by_category, link


def _region_has_content(
    providers: list[str], by_category: dict[str, list[str]], link: str | None
) -> bool:
    return bool(providers or by_category or link)


async def fetch_watch_info(
    client: Any,
    media_type: str,
    tmdb_id: int,
    watch_region: str | None,
    memo: WatchMemo | None = None,
) -> tuple[list[str], dict[str, list[str]], str | None, str]:
    region = effective_watch_region(watch_region)
    key = (media_type, tmdb_id, region)
    if memo is not None and key in memo:
        providers, by_category, link = memo[key]
        return providers, by_category, link, region
    candidates = [region] if region == "US" else [region, "US"]
    result: tuple[list[str], dict[str, list[str]], str | None] = ([], {}, None)
    used = region
    for candidate_region in candidates:
        candidate_key = (media_type, tmdb_id, candidate_region)
        if memo is not None and candidate_key in memo:
            providers, by_category, link = memo[candidate_key]
            if _region_has_content(providers, by_category, link) or candidate_region == "US":
                result = (providers, by_category, link)
                used = candidate_region
                break
            continue
        try:
            watch_data = await get_watch_providers(client, media_type, tmdb_id)
        except (TMDBError, httpx.HTTPError, ValueError, TypeError):
            providers, by_category, link = [], {}, None
        else:
            results = watch_data.get("results") if isinstance(watch_data, dict) else None
            region_data = results.get(candidate_region) if isinstance(results, dict) else None
            providers, by_category, link = _parse_region_providers(region_data)
        if memo is not None:
            memo[candidate_key] = (providers, by_category, link)
        if _region_has_content(providers, by_category, link) or candidate_region == "US":
            result = (providers, by_category, link)
            used = candidate_region
            break
    providers, by_category, link = result
    if memo is not None:
        memo[key] = (providers, by_category, link)
    return providers, by_category, link, used


async def to_candidate(
    client: Any,
    media_type: str,
    raw: dict[str, Any],
    watch_region: str | None = None,
    watch_memo: WatchMemo | None = None,
) -> Candidate:
    tmdb_id = raw.get("id") or raw.get("tmdb_id")
    if not isinstance(tmdb_id, int):
        raise ValueError("TMDB result is missing an integer id.")

    details = raw
    if not all(key in raw for key in ("credits", "keywords", "external_ids")):
        details = await get_details(client, media_type, tmdb_id)

    keyword_data = details.get("keywords") or {}
    if isinstance(keyword_data, list):
        keyword_items = keyword_data
    else:
        keyword_items = keyword_data.get("keywords") or keyword_data.get("results") or []
    keyword_ids = [item["id"] for item in keyword_items if isinstance(item, dict) and isinstance(item.get("id"), int)]

    credits = details.get("credits") or {}
    cast_ids = [
        item["id"]
        for item in credits.get("cast", [])[:20]
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    ]
    external_ids = details.get("external_ids") or {}
    release_date = details.get("release_date") or details.get("first_air_date")
    runtime = details.get("runtime")
    if runtime is None:
        episode_runtime = details.get("episode_run_time") or []
        runtime = episode_runtime[0] if episode_runtime else None

    providers: list[str] = []
    by_category: dict[str, list[str]] = {}
    watch_link: str | None = None
    effective_region = effective_watch_region(watch_region)
    try:
        providers, by_category, watch_link, effective_region = await fetch_watch_info(
            client, media_type, tmdb_id, watch_region, memo=watch_memo
        )
    except (TMDBError, httpx.HTTPError, ValueError, TypeError):
        providers, by_category, watch_link = [], {}, None

    title = details.get("title") or details.get("name") or raw.get("title") or raw.get("name")
    if not isinstance(title, str) or not title:
        raise ValueError(f"TMDB result {tmdb_id} is missing a title.")

    return Candidate(
        tmdb_id=tmdb_id,
        media_type=media_type,
        title=title,
        year=_year(release_date),
        genre_ids=[
            genre["id"]
            for genre in details.get("genres", [])
            if isinstance(genre, dict) and isinstance(genre.get("id"), int)
        ] or [genre_id for genre_id in details.get("genre_ids", []) if isinstance(genre_id, int)],
        keyword_ids=keyword_ids,
        cast_ids=cast_ids,
        vote_average=float(details.get("vote_average") or 0),
        vote_count=int(details.get("vote_count") or 0),
        runtime=runtime if isinstance(runtime, int) else None,
        original_language=details.get("original_language"),
        overview=str(details.get("overview") or ""),
        poster_path=details.get("poster_path"),
        imdb_id=external_ids.get("imdb_id") or details.get("imdb_id"),
        watch_providers=providers,
        watch_providers_by_category=by_category,
        watch_region=effective_region,
        watch_link=watch_link,
    )


async def enrich_with_omdb(candidate: Candidate) -> Candidate:
    if not candidate.imdb_id or not get_settings().omdb_api_key:
        return candidate
    try:
        ratings = await asyncio.to_thread(get_title_ratings, candidate.imdb_id)
    except OMDbError:
        return candidate

    imdb_votes_value = ratings.get("imdb_votes")
    try:
        imdb_votes = int(imdb_votes_value.replace(",", "")) if imdb_votes_value else None
    except ValueError:
        imdb_votes = None

    return candidate.model_copy(
        update={
            "imdb_rating": _rating_value(ratings.get("imdb_rating")),
            "imdb_votes": imdb_votes,
            "rt_score": _rating_value(ratings.get("rotten_tomatoes"), percent=True),
            "metacritic_score": _rating_value(ratings.get("metacritic"), percent=True),
        }
    )


async def rank_with_optional_omdb(
    pool: list[Candidate],
    source: Candidate | None,
    limit: int,
    *,
    min_vote_count: int | None = None,
    min_vote_average: float | None = None,
) -> list[ScoredCandidate]:
    from app.recommend.scoring import (
        MIN_AUDIENCE_SCORE,
        MIN_VOTE_COUNT,
        filter_and_rank,
    )

    effective_vote_count = (
        MIN_VOTE_COUNT if min_vote_count is None else min_vote_count
    )
    effective_vote_average = (
        MIN_AUDIENCE_SCORE if min_vote_average is None else min_vote_average
    )
    shortlist_ranked = filter_and_rank(
        pool,
        source=source,
        limit=min(max(limit * 2, limit), 10),
        min_vote_count=effective_vote_count,
        min_vote_average=effective_vote_average,
    )
    shortlist = [
        Candidate.model_validate(
            item.model_dump(exclude={"score", "reasons"})
        )
        for item in shortlist_ranked
    ]
    if get_settings().omdb_api_key and shortlist:
        shortlist = list(
            await asyncio.gather(*(enrich_with_omdb(item) for item in shortlist))
        )

    return filter_and_rank(
        shortlist,
        source=source,
        limit=limit,
        min_vote_count=effective_vote_count,
        min_vote_average=effective_vote_average,
    )
