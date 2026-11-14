"""Runtime configuration, read from the environment and the backend folder's .env file."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    project: str = Field("", validation_alias="GOOGLE_CLOUD_PROJECT")
    location: str = Field("global", validation_alias="GOOGLE_CLOUD_LOCATION")
    credentials_path: str | None = Field(None, validation_alias="GOOGLE_APPLICATION_CREDENTIALS")

    gemini_model: str = Field(
        "gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.8-flash", validation_alias="GEMINI_MODEL"
    )
    thinking_level: str | None = Field("minimal", validation_alias="GEMINI_THINKING_LEVEL")

    gemini_tts_model: str = Field(
        "gemini-3.8-flash-lite-tts,gemini-3.8-flash-tts", validation_alias="GEMINI_TTS_MODEL"
    )
    tts_location: str = Field("global", validation_alias="GEMINI_TTS_LOCATION")
    tts_voice: str = Field("Kore", validation_alias="GEMINI_TTS_VOICE")

    timeout_s: float = Field(30.0, validation_alias="LLM_TIMEOUT_S")
    tts_timeout_s: float = Field(45.0, validation_alias="TTS_TIMEOUT_S")

    rate_limit_per_minute: int = Field(20, validation_alias="RATE_LIMIT_PER_MINUTE")
    cache_ttl_s: int = Field(600, validation_alias="CACHE_TTL_S")
    cache_size: int = Field(256, validation_alias="CACHE_SIZE")
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")

    @field_validator("thinking_level", mode="before")
    @classmethod
    def _empty_thinking_is_none(cls, value: object) -> str | None:
        if value is None or str(value).strip() == "":
            return None
        return str(value).strip()

    @field_validator("credentials_path", mode="before")
    @classmethod
    def _resolve_credentials_path(cls, value: object) -> str | None:
        if value is None or str(value).strip() == "":
            return None
        path = Path(str(value).strip()).expanduser()
        if not path.is_absolute():
            path = BACKEND_DIR / path
        return str(path)

    @property
    def models(self) -> list[str]:
        """Ordered text-model fallbacks."""
        return _split_csv(self.gemini_model)

    @property
    def tts_models(self) -> list[str]:
        """Ordered TTS-model fallbacks."""
        return _split_csv(self.gemini_tts_model)

    def export_credentials(self) -> None:
        """Make the key file visible to google-auth, which only reads the process environment."""
        if self.credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.export_credentials()
    return settings
