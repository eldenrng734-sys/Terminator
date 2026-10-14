import pytest

from app.recommend.discover_params import build_discover_params


def test_movie_params_encode_and_or_filters_and_movie_dates():
    params = build_discover_params(
        "movie",
        genre_ids=[18, 53],
        keyword_ids_any=[9713, 818],
        year_from=1990,
        year_to=2000,
        runtime_min=90,
        runtime_max=140,
        language="en",
        watch_region="us",
        watch_provider_ids=[8, 9],
        min_vote_average=7.5,
        min_vote_count=500,
        page=2,
    )

    assert params == {
        "sort_by": "vote_average.desc",
        "vote_average.gte": 7.5,
        "vote_count.gte": 500,
        "include_adult": False,
        "page": 2,
        "with_genres": "18,53",
        "with_keywords": "9713|818",
        "primary_release_date.gte": "1990-01-01",
        "primary_release_date.lte": "2000-12-31",
        "with_runtime.gte": 90,
        "with_runtime.lte": 140,
        "with_original_language": "en",
        "with_watch_providers": "8|9",
        "watch_region": "US",
    }


def test_tv_uses_first_air_dates_and_omits_providers_without_region():
    params = build_discover_params(
        "tv", year_from=2020, watch_provider_ids=[8]
    )

    assert params["first_air_date.gte"] == "2020-01-01"
    assert "primary_release_date.gte" not in params
    assert "with_watch_providers" not in params
    assert "watch_region" not in params
    assert params["sort_by"] == "vote_average.desc"
    assert params["vote_average.gte"] == 7.0
    assert params["vote_count.gte"] == 200


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"page": 0}, "page"),
        ({"min_vote_average": 11}, "min_vote_average"),
        ({"min_vote_count": -1}, "min_vote_count"),
        ({"runtime_min": 120, "runtime_max": 90}, "runtime_min"),
        ({"year_from": 2025, "year_to": 2020}, "year_from"),
    ],
)
def test_invalid_discover_filters_raise_value_error(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_discover_params("movie", **kwargs)


def test_invalid_media_type_raises_value_error():
    with pytest.raises(ValueError, match="media_type"):
        build_discover_params("person")