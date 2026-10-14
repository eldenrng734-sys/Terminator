from typing import Literal

MediaType = Literal["movie", "tv"]
DiscoverValue = str | int | float | bool


def build_discover_params(
    media_type: MediaType,
    *,
    genre_ids: list[int] | None = None,
    keyword_ids_any: list[int] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    runtime_max: int | None = None,
    runtime_min: int | None = None,
    language: str | None = None,
    watch_region: str | None = None,
    watch_provider_ids: list[int] | None = None,
    min_vote_average: float = 7.0,
    min_vote_count: int = 200,
    page: int = 1,
) -> dict[str, DiscoverValue]:
    if media_type not in ("movie", "tv"):
        raise ValueError("media_type must be 'movie' or 'tv'.")
    if page < 1:
        raise ValueError("page must be at least 1.")
    if not 0 <= min_vote_average <= 10:
        raise ValueError("min_vote_average must be between 0 and 10.")
    if min_vote_count < 0:
        raise ValueError("min_vote_count cannot be negative.")
    if runtime_min is not None and runtime_max is not None and runtime_min > runtime_max:
        raise ValueError("runtime_min cannot exceed runtime_max.")
    if year_from is not None and year_to is not None and year_from > year_to:
        raise ValueError("year_from cannot be later than year_to.")

    date_field = "primary_release_date" if media_type == "movie" else "first_air_date"
    params: dict[str, DiscoverValue] = {
        "sort_by": "vote_average.desc",
        "vote_average.gte": min_vote_average,
        "vote_count.gte": min_vote_count,
        "include_adult": False,
        "page": page,
    }

    if genre_ids:
        params["with_genres"] = ",".join(map(str, genre_ids))
    if keyword_ids_any:
        params["with_keywords"] = "|".join(map(str, keyword_ids_any))
    if year_from is not None:
        params[f"{date_field}.gte"] = f"{year_from}-01-01"
    if year_to is not None:
        params[f"{date_field}.lte"] = f"{year_to}-12-31"
    if runtime_min is not None:
        params["with_runtime.gte"] = runtime_min
    if runtime_max is not None:
        params["with_runtime.lte"] = runtime_max
    if language:
        params["with_original_language"] = language
    if watch_provider_ids and watch_region:
        params["with_watch_providers"] = "|".join(map(str, watch_provider_ids))
        params["watch_region"] = watch_region.upper()

    return params