from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    groq_api_key: str
    openrouter_api_key: str
    llm_model: str = "google/gemma-4-26b-a4b-it:free"
    profile_name: str = "Chat user"
    plan_name: str = "API access"
    max_history_turns: int = 12
    max_image_mb: int = 8
    allowed_image_types: tuple[str, ...] = ("image/png", "image/jpeg", "image/webp")


@lru_cache
def get_settings() -> Settings:
    return Settings()