from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MediaType = Literal["movie", "tv"]
RequestedMediaType = Literal["movie", "tv", "both"]
RecommendationMode = Literal["similarity", "discovery", "off_topic"]


class Candidate(BaseModel):
    tmdb_id: int
    media_type: MediaType
    title: str
    year: int | None = None
    genre_ids: list[int] = Field(default_factory=list)
    keyword_ids: list[int] = Field(default_factory=list)
    cast_ids: list[int] = Field(default_factory=list)
    vote_average: float = 0.0
    vote_count: int = 0
    runtime: int | None = None
    original_language: str | None = None
    overview: str = ""
    poster_path: str | None = None
    imdb_id: str | None = None
    imdb_rating: float | None = None
    imdb_votes: int | None = None
    rt_score: int | None = None
    metacritic_score: int | None = None
    watch_providers: list[str] = Field(default_factory=list)
    watch_providers_by_category: dict[str, list[str]] = Field(default_factory=dict)
    watch_region: str | None = None
    watch_link: str | None = None


class ScoredCandidate(Candidate):
    score: float
    reasons: list[str] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    media_type: RequestedMediaType | None = None
    watch_region: str | None = Field(default=None, min_length=2, max_length=2)
    limit: int = Field(default=5, ge=1, le=10)


class RecommendationIntent(BaseModel):
    mode: RecommendationMode
    media_type: RequestedMediaType = "both"
    reference_title: str | None = None
    reference_year: int | None = Field(default=None, ge=1888, le=2100)
    genre_terms: list[str] = Field(default_factory=list)
    mood_terms: list[str] = Field(default_factory=list)
    year_from: int | None = Field(default=None, ge=1888, le=2100)
    year_to: int | None = Field(default=None, ge=1888, le=2100)
    runtime_min: int | None = Field(default=None, ge=1, le=1000)
    runtime_max: int | None = Field(default=None, ge=1, le=1000)
    watch_provider_terms: list[str] = Field(default_factory=list)
    language: str | None = Field(default=None, min_length=2, max_length=2)
    watch_region: str | None = Field(default=None, min_length=2, max_length=2)

    @field_validator("genre_terms", "mood_terms", "watch_provider_terms", mode="before")
    @classmethod
    def validate_term_lists(cls, value: object) -> list[str]:
        if not isinstance(value, list) or any(not isinstance(term, str) for term in value):
            raise ValueError("recommendation terms must be a list of strings")
        return value

    @model_validator(mode="after")
    def validate_ranges(self) -> RecommendationIntent:
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from cannot be later than year_to")
        if (
            self.runtime_min is not None
            and self.runtime_max is not None
            and self.runtime_min > self.runtime_max
        ):
            raise ValueError("runtime_min cannot exceed runtime_max")
        return self


class RecommendationResponse(BaseModel):
    mode: Literal["similarity", "discovery"]
    reply: str
    candidates: list[ScoredCandidate]