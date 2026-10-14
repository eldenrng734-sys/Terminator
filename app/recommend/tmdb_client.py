from collections.abc import Mapping
from typing import Any

import httpx

from app.config import get_settings
from app.services.cache import AsyncInMemoryTTLCache

TMDB_BASE_URL = "https://api.themoviedb.org/3"
_response_cache = AsyncInMemoryTTLCache(maxsize=2048, ttl_seconds=6 * 60 * 60)


class TMDBError(RuntimeError):
    pass


def create_client() -> httpx.AsyncClient:
    settings = get_settings()
    headers: dict[str, str] = {"Accept": "application/json"}
    if settings.tmdb_read_access_token:
        headers["Authorization"] = f"Bearer {settings.tmdb_read_access_token}"
    elif not settings.tmdb_api_key:
        raise TMDBError("TMDB_READ_ACCESS_TOKEN or TMDB_API_KEY is not configured.")
    return httpx.AsyncClient(base_url=TMDB_BASE_URL, headers=headers, timeout=10.0)


async def _get(client: httpx.AsyncClient, path: str, **params: Any) -> dict[str, Any]:
    settings = get_settings()
    request_params = dict(params)
    if not settings.tmdb_read_access_token:
        if not settings.tmdb_api_key:
            raise TMDBError("TMDB_READ_ACCESS_TOKEN or TMDB_API_KEY is not configured.")
        request_params["api_key"] = settings.tmdb_api_key

    cache_params = tuple(sorted((key, str(value)) for key, value in params.items()))
    cache_key = (path, cache_params)

    async def load() -> dict[str, Any]:
        try:
            response = await client.get(path, params=request_params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None:
                raise TMDBError(
                    f"TMDB returned HTTP {status} for {path}."
                ) from exc
            raise TMDBError(f"TMDB request failed for {path}.") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise TMDBError(f"TMDB returned invalid JSON for {path}.") from exc
        if not isinstance(data, dict):
            raise TMDBError(f"TMDB returned an unexpected response for {path}.")
        return data

    return await _response_cache.get_or_load(cache_key, load)


def _results(data: Mapping[str, Any], path: str) -> list[dict[str, Any]]:
    results = data.get("results", [])
    if not isinstance(results, list):
        raise TMDBError(f"TMDB returned invalid results for {path}.")
    return [item for item in results if isinstance(item, dict)]


def _validate_media_type(media_type: str) -> None:
    if media_type not in {"movie", "tv"}:
        raise ValueError("media_type must be 'movie' or 'tv'.")


async def get_similar(
    client: httpx.AsyncClient, media_type: str, tmdb_id: int
) -> list[dict[str, Any]]:
    _validate_media_type(media_type)
    path = f"/{media_type}/{tmdb_id}/similar"
    return _results(await _get(client, path), path)


async def get_recommendations(
    client: httpx.AsyncClient, media_type: str, tmdb_id: int
) -> list[dict[str, Any]]:
    _validate_media_type(media_type)
    path = f"/{media_type}/{tmdb_id}/recommendations"
    return _results(await _get(client, path), path)


async def get_details(
    client: httpx.AsyncClient, media_type: str, tmdb_id: int
) -> dict[str, Any]:
    _validate_media_type(media_type)
    return await _get(
        client,
        f"/{media_type}/{tmdb_id}",
        append_to_response="keywords,credits,external_ids",
    )


async def discover(
    client: httpx.AsyncClient, media_type: str, **filters: Any
) -> list[dict[str, Any]]:
    _validate_media_type(media_type)
    path = f"/discover/{media_type}"
    return _results(await _get(client, path, **filters), path)


async def search_title(
    client: httpx.AsyncClient,
    media_type: str,
    query: str,
    year: int | None = None,
) -> list[dict[str, Any]]:
    _validate_media_type(media_type)
    params: dict[str, Any] = {"query": query, "include_adult": False}
    if year is not None:
        params["year" if media_type == "movie" else "first_air_date_year"] = year
    path = f"/search/{media_type}"
    return _results(await _get(client, path, **params), path)


async def search_keyword(
    client: httpx.AsyncClient, query: str
) -> list[dict[str, Any]]:
    path = "/search/keyword"
    return _results(await _get(client, path, query=query), path)


async def get_genres(
    client: httpx.AsyncClient, media_type: str
) -> list[dict[str, Any]]:
    _validate_media_type(media_type)
    path = f"/genre/{media_type}/list"
    data = await _get(client, path, language="en-US")
    genres = data.get("genres", [])
    if not isinstance(genres, list):
        raise TMDBError(f"TMDB returned invalid genres for {path}.")
    return [item for item in genres if isinstance(item, dict)]


async def get_watch_providers(
    client: httpx.AsyncClient, media_type: str, tmdb_id: int
) -> dict[str, Any]:
    _validate_media_type(media_type)
    path = f"/{media_type}/{tmdb_id}/watch/providers"
    return await _get(client, path)


async def get_watch_provider_list(
    client: httpx.AsyncClient, media_type: str, watch_region: str
) -> list[dict[str, Any]]:
    _validate_media_type(media_type)
    path = f"/watch/providers/{media_type}"
    data = await _get(
        client,
        path,
        watch_region=watch_region.upper(),
        language="en-US",
    )
    return _results(data, path)