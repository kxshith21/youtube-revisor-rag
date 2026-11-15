"""API settings (pydantic-settings) - reads .env once per process."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path so the API works regardless of the current directory.
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    """Environment-driven configuration. Never stores secrets in code."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None

    # LLM
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "openai/gpt-oss-20b"
    GROQ_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None

    # API behaviour
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8501"
    API_KEY: str | None = None
    RATE_LIMIT_SEARCH: int = 30
    RATE_LIMIT_ASK: int = 10
    LOG_LEVEL: str = "INFO"
    ASK_TIMEOUT_S: float = 30.0
    STREAM_TIMEOUT_S: float = 60.0
    STREAM_HEARTBEAT_S: float = 15.0

    # In-memory caches (seconds / number of entries)
    CACHE_TTL_S: int = 600
    CACHE_MAX_ENTRIES: int = 500

    @field_validator(
        "QDRANT_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "API_KEY", mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """Treat empty env values (KEY=) as unset."""
        return value or None

    @property
    def origins(self) -> list[str]:
        """ALLOWED_ORIGINS as a list (comma-separated in .env)."""
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def llm_key_present(self) -> bool:
        """True when the configured provider has a usable key (ollama needs none)."""
        provider = (self.LLM_PROVIDER or "").lower()
        if provider == "groq":
            return bool(self.GROQ_API_KEY)
        if provider == "anthropic":
            return bool(self.ANTHROPIC_API_KEY)
        if provider in ("openai", "openai-compatible"):
            return bool(self.OPENAI_API_KEY)
        return True  # ollama / local endpoints may run without a key


@lru_cache
def get_settings() -> Settings:
    """Cached settings (env + .env read once)."""
    return Settings()
