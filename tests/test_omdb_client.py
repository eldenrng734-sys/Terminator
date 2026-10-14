from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from app.services import omdb_client
from app.services.cache import InMemoryTTLCache


@pytest.fixture(autouse=True)
def fresh_omdb_caches(monkeypatch):
    monkeypatch.setattr(
        omdb_client, "_ratings_cache", InMemoryTTLCache(512, 24 * 60 * 60)
    )
    monkeypatch.setattr(
        omdb_client, "_poster_cache", InMemoryTTLCache(16, 24 * 60 * 60)
    )


def test_get_title_ratings_extracts_requested_fields(monkeypatch):
    monkeypatch.setattr(
        omdb_client,
        "get_settings",
        lambda: SimpleNamespace(omdb_api_key="test-key"),
    )
    response = httpx.Response(
        200,
        json={
            "Response": "True",
            "imdbID": "tt1234567",
            "imdbRating": "8.2",
            "imdbVotes": "123,456",
            "Ratings": [
                {"Source": "Rotten Tomatoes", "Value": "94%"},
                {"Source": "Metacritic", "Value": "81/100"},
            ],
        },
        request=httpx.Request("GET", omdb_client.OMDB_DATA_URL),
    )
    get = Mock(return_value=response)
    monkeypatch.setattr(omdb_client.httpx, "get", get)

    result = omdb_client.get_title_ratings("tt1234567")

    assert result == {
        "imdb_id": "tt1234567",
        "imdb_rating": "8.2",
        "imdb_votes": "123,456",
        "rotten_tomatoes": "94%",
        "metacritic": "81/100",
    }
    assert get.call_args.args[0] == omdb_client.OMDB_DATA_URL
    assert get.call_args.kwargs["params"] == {
        "apikey": "test-key",
        "i": "tt1234567",
    }


def test_repeated_rating_lookup_uses_cache(monkeypatch):
    monkeypatch.setattr(
        omdb_client,
        "get_settings",
        lambda: SimpleNamespace(omdb_api_key="test-key"),
    )
    response = httpx.Response(
        200,
        json={"Response": "True", "imdbID": "tt7654321", "imdbRating": "7.5"},
        request=httpx.Request("GET", omdb_client.OMDB_DATA_URL),
    )
    get = Mock(return_value=response)
    monkeypatch.setattr(omdb_client.httpx, "get", get)

    first = omdb_client.get_title_ratings(" tt7654321 ")
    second = omdb_client.get_title_ratings("TT7654321")

    assert first == second
    assert get.call_count == 1


def test_missing_ratings_are_normalized_to_none(monkeypatch):
    monkeypatch.setattr(
        omdb_client,
        "get_settings",
        lambda: SimpleNamespace(omdb_api_key="test-key"),
    )
    response = httpx.Response(
        200,
        json={"Response": "True", "imdbRating": "N/A", "Ratings": []},
        request=httpx.Request("GET", omdb_client.OMDB_DATA_URL),
    )
    monkeypatch.setattr(omdb_client.httpx, "get", Mock(return_value=response))

    result = omdb_client.get_title_ratings("tt1234567")

    assert result["imdb_rating"] is None
    assert result["rotten_tomatoes"] is None
    assert result["metacritic"] is None


def test_omdb_error_response_is_reported(monkeypatch):
    monkeypatch.setattr(
        omdb_client,
        "get_settings",
        lambda: SimpleNamespace(omdb_api_key="test-key"),
    )
    response = httpx.Response(
        200,
        json={"Response": "False", "Error": "Invalid IMDb ID."},
        request=httpx.Request("GET", omdb_client.OMDB_DATA_URL),
    )
    monkeypatch.setattr(omdb_client.httpx, "get", Mock(return_value=response))

    with pytest.raises(omdb_client.OMDbError, match="Invalid IMDb ID"):
        omdb_client.get_title_ratings("tt1234567")


def test_fetch_poster_returns_image_bytes(monkeypatch):
    monkeypatch.setattr(
        omdb_client,
        "get_settings",
        lambda: SimpleNamespace(omdb_api_key="test-key"),
    )
    response = httpx.Response(
        200,
        content=b"image-data",
        headers={"content-type": "image/jpeg"},
        request=httpx.Request("GET", omdb_client.OMDB_POSTER_URL),
    )
    get = Mock(return_value=response)
    monkeypatch.setattr(omdb_client.httpx, "get", get)

    image, content_type = omdb_client.fetch_poster("tt1234567")

    assert image == b"image-data"
    assert content_type == "image/jpeg"
    assert get.call_args.args[0] == omdb_client.OMDB_POSTER_URL