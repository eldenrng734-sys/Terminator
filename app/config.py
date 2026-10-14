from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    groq_api_key: str
    openrouter_api_key: str
    google_client_id: str
    google_client_secret: str
    github_client_id: str
    github_client_secret: str
    tmdb_api_key: str | None = None
    tmdb_read_access_token: str | None = None
    omdb_api_key: str | None = None
    frontend_host: str = "http://localhost:8000"
    secret_key: str
    env: str = "development"
    llm_model: str = "google/gemma-4-26b-a4b-it:free"
    default_watch_region: str = "US"
    surprise_min_vote_average: float = 7.5
    profile_name: str = "Terminator user"
    plan_name: str = "API access"
    max_history_turns: int = 12
    max_image_mb: int = 8
    allowed_image_types: tuple[str, ...] = ("image/png", "image/jpeg", "image/webp")


@lru_cache
def get_settings() -> Settings:
    return Settings()