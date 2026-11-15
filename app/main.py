"""FastAPI application: lifespan, middleware, logging, and exception handling."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from generate import GenerationError, make_llm
from retrieve import RetrievalError, Retriever

from . import services
from .deps import (
    ApiError,
    AppState,
    MemoryCache,
    limiter,
    set_runtime_settings,
)
from .routers import ask, meta, revise, search
from .settings import Settings, get_settings

logger = logging.getLogger("app.main")
REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")
_UNSET = object()

_EXTRA_FIELDS = (
    "event",
    "method",
    "path",
    "status",
    "duration_ms",
    "request_id",
    "query_length",
    "result_count",
    "error_type",
)


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter with a fixed, secret-free field allowlist."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": REQUEST_ID.get(),
        }
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


_LOGGING_READY = False


def _configure_logging(level: str) -> None:
    """Install the JSON formatter once and apply the configured level."""
    global _LOGGING_READY
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not _LOGGING_READY:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
        _LOGGING_READY = True


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model, Qdrant client and LLM once; keep serving when degraded."""
    settings: Settings = app.state.settings
    retriever = app.state.retriever_override
    if retriever is None:
        try:
            retriever = Retriever()
        except Exception:
            logger.exception("retriever_init_failed")
            retriever = None

    embedding_ready = False
    if retriever is not None:
        try:
            await run_in_threadpool(retriever.embed_query, "warm up")
            embedding_ready = True
        except Exception:
            logger.exception("embedding_warmup_failed")

    if retriever is None:
        probe = services.ProbeResult(
            ok=False,
            reachable=False,
            points=0,
            detail="The retriever could not be initialized.",
        )
    else:
        probe = await run_in_threadpool(services.probe_index, retriever)

    if retriever is not None and embedding_ready and probe.ok:
        try:
            await run_in_threadpool(retriever.search, "warm up", top_k=1)
        except Exception:
            logger.exception("retrieval_warmup_failed")
            probe = services.ProbeResult(
                ok=False,
                reachable=probe.reachable,
                points=probe.points,
                detail="The retrieval warm-up query failed.",
            )

    llm: Any | None = None
    llm_detail: str | None = None
    llm_override = app.state.llm_override
    if llm_override is not _UNSET:
        llm = llm_override
        if llm is None:
            llm_detail = "No LLM client is configured for this application."
    elif not settings.llm_key_present:
        llm_detail = (
            f"LLM key is missing for provider '{settings.LLM_PROVIDER}'. "
            "Set the provider API key in .env; /search still works."
        )
        logger.error(
            "llm_key_missing",
            extra={"event": "startup", "error_type": "llm_key_missing"},
        )
    else:
        try:
            llm = await run_in_threadpool(make_llm)
        except Exception:
            logger.exception("llm_init_failed")
            llm_detail = (
                "The LLM client could not be initialized. Check LLM_PROVIDER, "
                "the provider package, and the API key."
            )

    details: list[str] = []
    if not embedding_ready:
        details.append("The embedding model failed to load.")
    if probe.detail:
        details.append(probe.detail)
    index_detail = " ".join(details) or None
    index_ok = bool(probe.ok and embedding_ready)

    app.state.resources = AppState(
        settings=settings,
        retriever=retriever,
        llm=llm,
        index_ok=index_ok,
        qdrant_reachable=probe.reachable,
        collection_points=probe.points,
        embedding_ready=embedding_ready,
        index_detail=index_detail,
        llm_detail=llm_detail,
        answer_cache=MemoryCache(
            ttl=settings.CACHE_TTL_S, maxsize=settings.CACHE_MAX_ENTRIES
        ),
        topics_cache=MemoryCache(
            ttl=settings.CACHE_TTL_S, maxsize=settings.CACHE_MAX_ENTRIES
        ),
        videos_cache=MemoryCache(
            ttl=settings.CACHE_TTL_S, maxsize=settings.CACHE_MAX_ENTRIES
        ),
    )
    logger.info(
        "startup_complete",
        extra={
            "event": "startup",
            "status": "ok" if index_ok and llm is not None else "degraded",
            "detail": index_detail or llm_detail,
        },
    )

    try:
        yield
    finally:
        if retriever is not None:
            client = getattr(retriever, "_client", None)
            if client is not None:
                try:
                    client.close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass


# --------------------------------------------------------------------------- #
# Error helpers
# --------------------------------------------------------------------------- #
def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None) or REQUEST_ID.get()
    return rid if rid and rid != "-" else uuid.uuid4().hex


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build the standard error envelope and tag it with the request ID."""
    rid = _request_id(request)
    response = JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": rid}},
        headers=headers,
    )
    response.headers["X-Request-ID"] = rid
    return response


def _validation_message(exc: RequestValidationError) -> str:
    """Summarize validation errors without echoing full user input."""
    parts: list[str] = []
    for error in exc.errors()[:3]:
        location = ".".join(
            str(part)
            for part in error.get("loc", ())
            if part not in ("body", "query", "path")
        )
        message = str(error.get("msg", "invalid value"))
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) or "Invalid request."


_HTTP_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    408: "timeout",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    429: "rate_limited",
}


def _http_message(status_code: int) -> str:
    return {
        400: "The request could not be processed.",
        401: "Authentication is required.",
        403: "You do not have permission to access this resource.",
        404: "The requested endpoint was not found.",
        405: "That HTTP method is not allowed for this endpoint.",
        429: "Too many requests. Try again shortly.",
    }.get(status_code, "The request failed.")


def _register_exception_handlers(app: FastAPI) -> None:
    """Attach the global error envelope handlers."""

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(request, exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            request, 422, "validation_error", _validation_message(exc)
        )

    @app.exception_handler(RateLimitExceeded)
    async def handle_rate_limit(
        request: Request, exc: RateLimitExceeded
    ) -> JSONResponse:
        return _error_response(
            request,
            429,
            "rate_limited",
            "Rate limit exceeded. Please retry later.",
            headers={"Retry-After": "60"},
        )

    @app.exception_handler(RetrievalError)
    async def handle_retrieval(
        request: Request, exc: RetrievalError
    ) -> JSONResponse:
        logger.warning(
            "retrieval_error",
            extra={"event": "error", "error_type": type(exc).__name__},
        )
        return _error_response(
            request,
            503,
            "index_unavailable",
            str(exc) or "The search index is unavailable.",
        )

    @app.exception_handler(services.QuizParseError)
    async def handle_quiz_parse(
        request: Request, exc: services.QuizParseError
    ) -> JSONResponse:
        logger.warning(
            "quiz_parse_error",
            extra={"event": "error", "error_type": type(exc).__name__},
        )
        return _error_response(request, 502, "quiz_parse_error", str(exc))

    @app.exception_handler(GenerationError)
    async def handle_generation(
        request: Request, exc: GenerationError
    ) -> JSONResponse:
        logger.exception(
            "generation_error",
            extra={"event": "error", "error_type": type(exc).__name__},
        )
        if services.is_llm_rate_limit(exc):
            return _error_response(
                request,
                429,
                "llm_rate_limited",
                "The language model is rate limited. Please retry shortly.",
                headers={"Retry-After": "30"},
            )
        if "valid json" in str(exc).lower():
            return _error_response(
                request,
                502,
                "quiz_parse_error",
                "The model did not return valid quiz JSON after a retry. Please try again.",
            )
        return _error_response(
            request,
            502,
            "llm_failed",
            "The language model request failed. Please retry shortly.",
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        status = exc.status_code
        detail = exc.detail
        message = (
            str(detail)
            if isinstance(detail, str) and detail
            else _http_message(status)
        )
        return _error_response(
            request,
            status,
            _HTTP_CODES.get(status, "http_error"),
            message,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled_error",
            extra={
                "event": "error",
                "error_type": type(exc).__name__,
                "path": request.url.path,
            },
        )
        return _error_response(
            request,
            500,
            "internal_error",
            "An unexpected error occurred. Please retry or report the request ID.",
        )


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def create_app(
    settings: Settings | None = None,
    *,
    retriever: Any | None = None,
    llm: Any = _UNSET,
) -> FastAPI:
    """Build the API. ``retriever``/``llm`` overrides exist for tests."""
    settings = settings or get_settings()
    set_runtime_settings(settings)
    _configure_logging(settings.LOG_LEVEL)

    app = FastAPI(
        title="YouTube Reviser API",
        version="1.0.0",
        description=(
            "RAG API over indexed Padho with Pratyush DSA and Gaurav Sen system design "
            "videos. Retrieval and generation are provided by the existing "
            "scripts/retrieve.py and scripts/generate.py modules.\n\n"
            "When API_KEY is configured, send it as the X-API-Key header. "
            "/health is always public. Interactive docs are available at /docs."
        ),
        lifespan=lifespan,
        openapi_tags=[
            {"name": "search", "description": "Qdrant retrieval only; no LLM."},
            {"name": "ask", "description": "Full RAG answers and SSE streaming."},
            {"name": "revise", "description": "Revision notes and quizzes."},
            {"name": "meta", "description": "Health and index metadata."},
        ],
    )
    app.state.settings = settings
    app.state.retriever_override = retriever
    app.state.llm_override = llm
    app.state.resources = None
    app.state.limiter = limiter

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # Retry-After must be exposed or browsers cannot read it on 429/503.
        expose_headers=["X-Request-ID", "X-Cache", "Retry-After"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        rid = uuid.uuid4().hex
        request.state.request_id = rid
        token = REQUEST_ID.set(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_failed",
                extra={
                    "event": "request_error",
                    "method": request.method,
                    "path": request.url.path,
                    "request_id": rid,
                },
            )
            raise
        else:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            response.headers["X-Request-ID"] = rid
            logger.info(
                "request_complete",
                extra={
                    "event": "request",
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                    "request_id": rid,
                    "query_length": getattr(request.state, "query_length", None),
                    "result_count": getattr(request.state, "result_count", None),
                },
            )
            return response
        finally:
            REQUEST_ID.reset(token)

    _register_exception_handlers(app)
    app.include_router(search.router)
    app.include_router(ask.router)
    app.include_router(revise.router)
    app.include_router(meta.router)
    return app


app = create_app()
