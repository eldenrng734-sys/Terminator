from typing import Any

import httpx

from app.config import get_settings
from app.services.cache import InMemoryTTLCache

OMDB_DATA_URL = "https://www.omdbapi.com/"
OMDB_POSTER_URL = "https://img.omdbapi.com/"
_ratings_cache = InMemoryTTLCache(maxsize=512, ttl_seconds=24 * 60 * 60)
_poster_cache = InMemoryTTLCache(maxsize=16, ttl_seconds=24 * 60 * 60)


class OMDbError(RuntimeError):
    pass


def _api_key() -> str:
    key = get_settings().omdb_api_key
    if not key:
        raise OMDbError("OMDB_API_KEY is not configured.")
    return key


def _get(url: str, imdb_id: str, timeout: float) -> httpx.Response:
    try:
        response = httpx.get(
            url,
            params={"apikey": _api_key(), "i": imdb_id},
            timeout=timeout,
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        raise OMDbError(f"OMDb request failed: {exc}") from exc

    if response.is_error:
        raise OMDbError(f"OMDb request failed with HTTP {response.status_code}.")
    return response


def _available(value: Any) -> str | None:
    if not isinstance(value, str) or not value or value == "N/A":
        return None
    return value


def _fetch_title_ratings(
    imdb_id: str, timeout: float
) -> dict[str, str | None]:
    response = _get(OMDB_DATA_URL, imdb_id, timeout)
    try:
        payload = response.json()
    except ValueError as exc:
        raise OMDbError("OMDb returned invalid JSON.") from exc

    if not isinstance(payload, dict) or payload.get("Response") != "True":
        message = payload.get("Error") if isinstance(payload, dict) else None
        raise OMDbError(message or "OMDb could not find this IMDb ID.")

    ratings = payload.get("Ratings")
    ratings_by_source = {
        item.get("Source"): item.get("Value")
        for item in ratings
        if isinstance(item, dict)
    } if isinstance(ratings, list) else {}

    return {
        "imdb_id": _available(payload.get("imdbID")) or imdb_id,
        "imdb_rating": _available(payload.get("imdbRating")),
        "imdb_votes": _available(payload.get("imdbVotes")),
        "rotten_tomatoes": _available(ratings_by_source.get("Rotten Tomatoes")),
        "metacritic": _available(ratings_by_source.get("Metacritic")),
    }


def get_title_ratings(imdb_id: str, timeout: float = 10.0) -> dict[str, str | None]:
    normalized_id = imdb_id.strip()
    if not normalized_id:
        raise OMDbError("An IMDb ID is required.")
    return _ratings_cache.get_or_load(
        normalized_id.casefold(),
        lambda: _fetch_title_ratings(normalized_id, timeout),
    )


def fetch_poster(imdb_id: str, timeout: float = 10.0) -> tuple[bytes, str]:
    normalized_id = imdb_id.strip()
    if not normalized_id:
        raise OMDbError("An IMDb ID is required.")
    return _poster_cache.get_or_load(
        normalized_id.casefold(),
        lambda: _fetch_poster(normalized_id, timeout),
    )


def _fetch_poster(imdb_id: str, timeout: float) -> tuple[bytes, str]:
    response = _get(OMDB_POSTER_URL, imdb_id, timeout)
    content_type = response.headers.get("content-type", "").split(";", 1)[0]
    if not content_type.startswith("image/"):
        raise OMDbError("OMDb did not return an image for this IMDb ID.")
    return response.content, content_type