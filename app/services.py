"""Thin service layer over ``scripts/retrieve.py`` and ``scripts/generate.py``.

Everything here is orchestration: timing, per-request domain injection,
response shaping, metadata queries, caching, and SSE formatting. Retrieval and
generation algorithms stay in the existing pipeline modules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator

from pydantic import ValidationError
from qdrant_client.http import models as qm

from scripts.config import COLLECTION_NAME, MODEL_NAME, VECTOR_SIZE
from scripts.generate import (  # noqa: E402  (scripts dir added in app/__init__.py)
    NO_RESULTS_MESSAGE,
    Answer,
    GenerationError,
    generate,
    generate_stream,
)
from scripts.retrieve import RetrievalError, SearchResult  # noqa: E402

from .deps import AppState, MemoryCache
from .schemas import (
    AnswerOut,
    HealthOut,
    QuizQuestion,
    QuizResponse,
    ReviseResponse,
    SourceOut,
    TopicOut,
    VideoOut,
)

logger = logging.getLogger("app.services")

DOMAINS = ("dsa", "system_design")
MAX_SCROLL_PAGES = 10_000


class QuizParseError(Exception):
    """The LLM returned JSON that could not be shaped into quiz questions."""


# --------------------------------------------------------------------------- #
# Retrieval wrapper + domain injection
# --------------------------------------------------------------------------- #
def _run_search(retriever: Any, query: str, **kwargs: Any) -> list[SearchResult]:
    """Call Retriever.search, translating backend failures into RetrievalError."""
    try:
        return retriever.search(query, **kwargs)
    except (ValueError, RetrievalError):
        raise
    except Exception as exc:  # Qdrant client/network/provider failure
        logger.exception("retrieval_failed")
        raise RetrievalError("Vector search is temporarily unavailable.") from exc


class DomainScopedRetriever:
    """Per-request adapter that injects ``domain`` into the shared Retriever.

    ``generate()`` does not accept a domain argument, so the API supplies this
    tiny proxy instead of monkeypatching the shared Retriever instance.
    """

    def __init__(self, retriever: Any, domain: str | None) -> None:
        self._retriever = retriever
        self.domain = domain
        self.last_results: list[SearchResult] = []

    @property
    def last_timings(self) -> dict[str, float]:
        """Expose the shared Retriever's last timing dict."""
        return getattr(
            self._retriever, "last_timings", {"embed_ms": 0.0, "search_ms": 0.0}
        )

    def search(self, query: str, **kwargs: Any) -> list[SearchResult]:
        """Run retrieval with this request's domain filter applied."""
        kwargs.setdefault("domain", self.domain)
        results = _run_search(self._retriever, query, **kwargs)
        self.last_results = results
        return results

    def __getattr__(self, name: str) -> Any:
        return getattr(self._retriever, name)


def search(
    retriever: Any,
    query: str,
    domain: str | None,
    top_k: int,
    use_mmr: bool,
    use_reranker: bool,
) -> tuple[list[SearchResult], float]:
    """Retrieval-only call used by POST /search."""
    started = time.perf_counter()
    results = _run_search(
        retriever,
        query,
        domain=domain,
        top_k=top_k,
        use_mmr=use_mmr,
        use_reranker=use_reranker,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    return results, latency_ms


# --------------------------------------------------------------------------- #
# Ask / revise / quiz
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AskResult:
    """Generated answer plus how many chunks retrieval returned."""

    answer: Answer
    result_count: int


@dataclass(frozen=True)
class CachedAnswer:
    """Cached /ask response and its retrieval count."""

    answer: AnswerOut
    result_count: int


def answer_to_out(answer: Answer) -> AnswerOut:
    """Validate a generate.Answer dataclass into the API response model."""
    return AnswerOut.model_validate(answer.to_dict())


def _sources(raw: list[dict[str, Any]]) -> list[SourceOut]:
    """Validate source dicts built in generate.py."""
    return [SourceOut.model_validate(source) for source in raw]


def ask(
    retriever: Any,
    llm: Any,
    query: str,
    domain: str | None,
    top_k: int,
    history: list[dict[str, str]] | None = None,
) -> AskResult:
    """Run the full RAG answer pipeline for one request."""
    adapter = DomainScopedRetriever(retriever, domain)
    answer = generate(
        query,
        retriever=adapter,
        llm=llm,
        top_k=top_k,
        mode="answer",
        history=history or None,
    )
    return AskResult(answer=answer, result_count=len(adapter.last_results))


def ask_cache_key(query: str, domain: str | None, top_k: int) -> tuple[str, str, int]:
    """Normalized key for the /ask response cache."""
    normalized = re.sub(r"\s+", " ", query or "").strip().casefold()
    return normalized, domain or "", top_k


def ask_with_cache(
    retriever: Any,
    llm: Any,
    query: str,
    domain: str | None,
    top_k: int,
    history: list[dict[str, str]],
    cache: MemoryCache,
) -> tuple[AnswerOut, str, int]:
    """Answer with lookup/store around the cache; returns (answer, HIT/MISS, count)."""
    cacheable = not history
    key = ask_cache_key(query, domain, top_k)
    if cacheable:
        cached = cache.get(key)
        if isinstance(cached, CachedAnswer):
            refreshed = cached.answer.model_copy(update={"query": query})
            return refreshed, "HIT", cached.result_count

    result = ask(retriever, llm, query, domain, top_k, history or None)
    out = answer_to_out(result.answer)
    if cacheable:
        cache.set(key, CachedAnswer(answer=out, result_count=result.result_count))
    return out, "MISS", result.result_count


def revise(retriever: Any, llm: Any, topic: str, domain: str | None) -> ReviseResponse:
    """Run generate(mode='revise') and shape the response."""
    adapter = DomainScopedRetriever(retriever, domain)
    answer = generate(topic, retriever=adapter, llm=llm, mode="revise")
    return ReviseResponse(
        topic=topic,
        note=answer.answer,
        sources=_sources(answer.sources),
        found=answer.found,
        uncited=answer.uncited,
        model=answer.model,
        latency_ms=answer.latency_ms,
        usage=answer.usage,
    )


def quiz(
    retriever: Any,
    llm: Any,
    topic: str,
    domain: str | None,
    num_questions: int,
) -> QuizResponse:
    """Run generate(mode='quiz') and parse its JSON question list."""
    adapter = DomainScopedRetriever(retriever, domain)
    answer = generate(
        topic,
        retriever=adapter,
        llm=llm,
        mode="quiz",
        num_questions=num_questions,
    )
    sources = _sources(answer.sources)
    if not answer.found or answer.answer == NO_RESULTS_MESSAGE:
        return QuizResponse(
            questions=[],
            sources=sources,
            found=False,
            model=answer.model,
            latency_ms=answer.latency_ms,
            usage=answer.usage,
        )

    try:
        data = json.loads(answer.answer)
        raw_questions = data["questions"]
        if not isinstance(raw_questions, list):
            raise TypeError("questions must be a list")
        questions = [QuizQuestion.model_validate(item) for item in raw_questions]
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        raise QuizParseError(
            "The model did not return valid quiz JSON after a retry. Please try again."
        ) from exc

    return QuizResponse(
        questions=questions,
        sources=sources,
        found=True,
        model=answer.model,
        latency_ms=answer.latency_ms,
        usage=answer.usage,
    )


class AnswerStream:
    """Streaming session wrapper; also records the retrieval result count."""

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        query: str,
        domain: str | None,
        top_k: int,
        history: list[dict[str, str]] | None = None,
    ) -> None:
        self._adapter = DomainScopedRetriever(retriever, domain)
        self._llm = llm
        self._query = query
        self._top_k = top_k
        self._history = history or None

    @property
    def result_count(self) -> int:
        """Number of chunks returned by retrieval for this stream."""
        return len(self._adapter.last_results)

    def __iter__(self) -> Iterator[str | Answer]:
        return generate_stream(
            self._query,
            retriever=self._adapter,
            llm=self._llm,
            top_k=self._top_k,
            mode="answer",
            history=self._history,
        )


# --------------------------------------------------------------------------- #
# Qdrant readiness + metadata
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProbeResult:
    """Result of the startup Qdrant/collection check."""

    ok: bool
    reachable: bool
    points: int
    detail: str | None


_EMBED_HINT = "Run first: python scripts/embed_and_store.py --domain all"


def probe_index(retriever: Any) -> ProbeResult:
    """Check Qdrant connectivity and that the collection is usable."""
    try:
        client = retriever.client
    except RetrievalError as exc:
        return ProbeResult(False, False, 0, str(exc))
    except Exception:  # pragma: no cover - defensive
        logger.exception("qdrant_client_failed")
        return ProbeResult(False, False, 0, "Qdrant is not reachable.")

    try:
        client.get_collections()
    except Exception:
        logger.exception("qdrant_unreachable")
        return ProbeResult(False, False, 0, "Qdrant is not reachable.")

    try:
        if not client.collection_exists(COLLECTION_NAME):
            return ProbeResult(
                False,
                True,
                0,
                f"Collection '{COLLECTION_NAME}' does not exist. {_EMBED_HINT}",
            )
        info = client.get_collection(COLLECTION_NAME)
        points = int(getattr(info, "points_count", 0) or 0)
        if points == 0:
            return ProbeResult(
                False,
                True,
                0,
                f"Collection '{COLLECTION_NAME}' is empty. {_EMBED_HINT}",
            )
        vectors = getattr(getattr(info.config, "params", None), "vectors", None)
        size = getattr(vectors, "size", None)
        if size is not None and size != VECTOR_SIZE:
            return ProbeResult(
                False,
                True,
                points,
                f"Collection vector size {size} != {VECTOR_SIZE}. "
                "Rebuild with: python scripts/embed_and_store.py --rebuild",
            )
        return ProbeResult(True, True, points, None)
    except Exception:
        logger.exception("qdrant_collection_probe_failed")
        return ProbeResult(False, True, 0, "Could not inspect the Qdrant collection.")


def _client(retriever: Any) -> Any:
    """Return the Qdrant client, translating connection failures to 503."""
    try:
        return retriever.client
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError("Qdrant is not reachable.") from exc


def _domain_filter(domain: str | None) -> qm.Filter | None:
    """Build a Qdrant filter for an optional domain."""
    if not domain:
        return None
    return qm.Filter(
        must=[qm.FieldCondition(key="domain", match=qm.MatchValue(value=domain))]
    )


def _collect_videos(client: Any, domain: str | None) -> list[VideoOut]:
    """Scroll the collection and aggregate chunks into distinct videos."""
    scroll_filter = _domain_filter(domain)
    videos: dict[str, dict[str, Any]] = {}
    offset: Any = None
    for _ in range(MAX_SCROLL_PAGES):
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=["video_id", "title", "channel", "domain"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            video_id = str(payload.get("video_id", ""))
            if not video_id:
                continue
            entry = videos.get(video_id)
            if entry is None:
                entry = {
                    "video_id": video_id,
                    "title": str(payload.get("title", "")),
                    "channel": str(payload.get("channel", "")),
                    "domain": str(payload.get("domain", "")),
                    "chunks": 0,
                }
                videos[video_id] = entry
            entry["chunks"] += 1
        if offset is None:
            break
    return [
        VideoOut(**entry)
        for entry in sorted(
            videos.values(),
            key=lambda item: (item["title"].casefold(), item["video_id"]),
        )
    ]


def topics(retriever: Any, cache: MemoryCache) -> list[TopicOut]:
    """Return per-domain video and chunk counts (cached by the caller)."""
    cached = cache.get("topics")
    if cached is not None:
        return cached

    client = _client(retriever)
    try:
        result: list[TopicOut] = []
        for domain in DOMAINS:
            count = client.count(
                collection_name=COLLECTION_NAME,
                count_filter=_domain_filter(domain),
                exact=True,
            )
            chunks = int(getattr(count, "count", 0) or 0)
            if chunks == 0:
                continue
            videos = _collect_videos(client, domain)
            result.append(
                TopicOut(domain=domain, videos=len(videos), chunks=chunks)
            )
    except RetrievalError:
        raise
    except Exception as exc:
        logger.exception("topics_failed")
        raise RetrievalError("Could not read topic metadata from Qdrant.") from exc

    cache.set("topics", result)
    return result


def videos(
    retriever: Any,
    cache: MemoryCache,
    domain: str | None,
    query: str | None,
    limit: int,
    offset: int,
) -> tuple[list[VideoOut], int]:
    """Return a filtered/paginated slice of distinct indexed videos."""
    key = domain or "all"
    cached = cache.get(key)
    if cached is None:
        client = _client(retriever)
        try:
            cached = _collect_videos(client, domain)
        except RetrievalError:
            raise
        except Exception as exc:
            logger.exception("videos_failed")
            raise RetrievalError("Could not read video metadata from Qdrant.") from exc
        cache.set(key, cached)

    items: list[VideoOut] = cached
    if query:
        needle = query.casefold()
        items = [item for item in items if needle in item.title.casefold()]
    total = len(items)
    return items[offset : offset + limit], total


def health(state: AppState) -> HealthOut:
    """Build the public health payload from startup state (no secrets)."""
    details = [state.index_detail, state.llm_detail]
    is_ok = (
        state.index_ok
        and state.embedding_ready
        and state.llm is not None
    )
    return HealthOut(
        status="ok" if is_ok else "degraded",
        qdrant_reachable=state.qdrant_reachable,
        collection_points=state.collection_points,
        embedding_model=MODEL_NAME,
        llm_provider=state.settings.LLM_PROVIDER,
        llm_model=(
            getattr(state.llm, "model", None)
            or state.settings.LLM_MODEL
        ),
        llm_ready=state.llm is not None,
        detail=" ".join(part for part in details if part) or None,
    )


# --------------------------------------------------------------------------- #
# SSE streaming
# --------------------------------------------------------------------------- #
def is_llm_rate_limit(exc: BaseException) -> bool:
    """Best-effort detection of provider 429s wrapped by generate.GenerationError."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "status_code", None) == 429:
            return True
        text = str(current).lower()
        if any(
            marker in text
            for marker in ("429", "rate limit", "rate_limit", "too many requests")
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _sse(event: str, data: dict[str, Any]) -> str:
    """Serialize one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_error_message(exc: BaseException) -> str:
    """Safe, user-facing message for a mid-stream failure."""
    if isinstance(exc, RetrievalError):
        return "The search index is unavailable. Please try again later."
    if isinstance(exc, GenerationError):
        if is_llm_rate_limit(exc):
            return "The language model is rate limited. Please try again shortly."
        return "The language model request failed. Please try again."
    return "Generation failed. Please try again."


def _stream_done_payload(answer: Answer) -> dict[str, Any]:
    """Validate the final Answer before sending the done event."""
    return answer_to_out(answer).model_dump(mode="json")


async def sse_events(
    session: AnswerStream,
    *,
    heartbeat_s: float,
    timeout_s: float,
    request_id: str,
) -> AsyncIterator[str]:
    """Yield SSE frames from a blocking generate_stream(), with heartbeats.

    A producer thread runs the synchronous generator and pushes items into an
    asyncio queue. The consumer owns the 15-second heartbeat and the overall
    deadline. Closing/cancelling this async generator sets a stop flag so the
    producer stops advancing at the next chunk.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    stop = threading.Event()
    started = time.perf_counter()
    heartbeat = max(0.05, float(heartbeat_s))
    deadline = time.monotonic() + max(0.1, float(timeout_s))

    def _put(item: tuple[str, Any]) -> None:
        try:
            queue.put_nowait(item)
        except RuntimeError:  # pragma: no cover - loop shut down
            pass

    def _emit(kind: str, payload: Any) -> None:
        try:
            loop.call_soon_threadsafe(_put, (kind, payload))
        except RuntimeError:  # pragma: no cover - loop shut down
            pass

    def _produce() -> None:
        try:
            iterator = iter(session)
            while not stop.is_set():
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                if stop.is_set():
                    break
                if isinstance(item, Answer):
                    _emit("done", _stream_done_payload(item))
                else:
                    _emit("token", {"text": str(item)})
            _emit("end", None)
        except BaseException as exc:  # noqa: BLE001 - stream must report, not crash
            if isinstance(exc, (GeneratorExit, KeyboardInterrupt, SystemExit)):
                raise
            logger.exception("stream_generation_failed", extra={"request_id": request_id})
            if not stop.is_set():
                _emit("error", {"message": _stream_error_message(exc)})
                _emit("end", None)

    worker = threading.Thread(target=_produce, name="ask-stream", daemon=True)
    worker.start()

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stop.set()
                yield _sse("error", {"message": "Generation timed out. Please retry."})
                break
            try:
                kind, payload = await asyncio.wait_for(
                    queue.get(), timeout=min(heartbeat, remaining)
                )
            except asyncio.TimeoutError:
                if time.monotonic() >= deadline:
                    stop.set()
                    yield _sse(
                        "error", {"message": "Generation timed out. Please retry."}
                    )
                    break
                yield ": heartbeat\n\n"
                continue

            if kind == "token":
                yield _sse("token", payload)
            elif kind == "done":
                yield _sse("done", payload)
                break
            elif kind == "error":
                yield _sse("error", payload)
                break
            elif kind == "end":
                break
    finally:
        stop.set()
        logger.info(
            "stream_complete",
            extra={
                "event": "stream_complete",
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "result_count": session.result_count,
                "request_id": request_id,
            },
        )
