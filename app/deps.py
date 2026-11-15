"""Shared application state, caches, API errors, and FastAPI dependencies.

The embedding model and Qdrant client live inside one ``Retriever`` instance
created at startup. The LLM client is created once too. Requests only borrow
those objects; nothing here reimplements retrieval or generation logic.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from typing import Annotated, Any

from cachetools import TTLCache
from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from .schemas import ErrorEnvelope
from .settings import Settings, get_settings


class ApiError(Exception):
    """HTTP-safe API failure returned in the standard error envelope."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class MemoryCache:
    """Small thread-safe TTL cache (used for /ask, /topics and /videos)."""

    def __init__(self, ttl: float = 600, maxsize: int = 500) -> None:
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any:
        """Return a cached value or None when missing/expired."""
        with self._lock:
            return self._cache.get(key)

    def set(self, key: Any, value: Any) -> None:
        """Store a value, evicting the least recently used entry if full."""
        with self._lock:
            self._cache[key] = value

    def clear(self) -> None:
        """Drop every cached entry."""
        with self._lock:
            self._cache.clear()


@dataclass
class AppState:
    """Resources and readiness flags created once during application startup."""

    settings: Settings
    retriever: Any | None
    llm: Any | None
    index_ok: bool
    qdrant_reachable: bool
    collection_points: int
    embedding_ready: bool
    index_detail: str | None
    llm_detail: str | None
    answer_cache: MemoryCache
    topics_cache: MemoryCache
    videos_cache: MemoryCache


# --------------------------------------------------------------------------- #
# Runtime settings / rate-limit callables
# --------------------------------------------------------------------------- #
_runtime_settings: Settings | None = None


def set_runtime_settings(settings: Settings) -> None:
    """Make the active app settings visible to slowapi limit callables."""
    global _runtime_settings
    _runtime_settings = settings


def get_runtime_settings() -> Settings:
    """Return the active settings, falling back to the .env-backed singleton."""
    return _runtime_settings or get_settings()


def search_rate_limit() -> str:
    """slowapi limit string for /search."""
    return f"{get_runtime_settings().RATE_LIMIT_SEARCH}/minute"


def ask_rate_limit() -> str:
    """slowapi limit string for LLM-backed endpoints."""
    return f"{get_runtime_settings().RATE_LIMIT_ASK}/minute"


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    headers_enabled=False,
)


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
def get_state(request: Request) -> AppState:
    """Return the startup-created state, or 503 while the app is starting."""
    state = getattr(request.app.state, "resources", None)
    if state is None:
        raise ApiError(503, "not_ready", "The API is still starting up. Try again shortly.")
    return state


def require_index(state: Annotated[AppState, Depends(get_state)]) -> None:
    """Reject requests that need Qdrant when the index is unavailable."""
    if not state.index_ok:
        raise ApiError(
            503,
            "index_unavailable",
            state.index_detail or "The search index is unavailable. Try again later.",
        )


def require_llm(state: Annotated[AppState, Depends(get_state)]) -> None:
    """Reject requests that need an LLM when no client is configured."""
    if state.llm is None:
        raise ApiError(
            503,
            "llm_unavailable",
            state.llm_detail or "The language model is not configured on this server.",
        )


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Required only when API_KEY is set on the server.",
)


def verify_api_key(
    state: Annotated[AppState, Depends(get_state)],
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> None:
    """Enforce X-API-Key when API_KEY is configured; otherwise auth is off."""
    expected = state.settings.API_KEY
    if not expected:
        return
    if not api_key:
        raise ApiError(401, "unauthorized", "Missing X-API-Key header.")
    if not secrets.compare_digest(api_key, expected):
        raise ApiError(401, "unauthorized", "Invalid API key.")


# --------------------------------------------------------------------------- #
# OpenAPI error responses reused by routers
# --------------------------------------------------------------------------- #
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorEnvelope, "description": "Missing or invalid API key."},
    422: {"model": ErrorEnvelope, "description": "Request validation failed."},
    429: {"model": ErrorEnvelope, "description": "Rate limit exceeded."},
    500: {"model": ErrorEnvelope, "description": "Unexpected server error."},
    502: {"model": ErrorEnvelope, "description": "The language model call failed."},
    503: {"model": ErrorEnvelope, "description": "Qdrant, the index, or the LLM is unavailable."},
    504: {"model": ErrorEnvelope, "description": "The request timed out."},
}
