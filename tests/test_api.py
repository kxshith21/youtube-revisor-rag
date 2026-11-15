"""API tests: fake Retriever/Qdrant/LLM, no model downloads or network."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.deps import limiter  # noqa: E402
from app.main import create_app  # noqa: E402
from app.settings import Settings  # noqa: E402
from retrieve import RetrievalError, SearchResult  # noqa: E402


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
def make_settings(**overrides: Any) -> Settings:
    """Settings isolated from the real .env, with fast test defaults."""
    values: dict[str, Any] = {
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": None,
        "LLM_PROVIDER": "groq",
        "LLM_MODEL": "test/model",
        "GROQ_API_KEY": None,
        "ANTHROPIC_API_KEY": None,
        "OPENAI_API_KEY": None,
        "ALLOWED_ORIGINS": "http://localhost:3000",
        "API_KEY": None,
        "RATE_LIMIT_SEARCH": 30,
        "RATE_LIMIT_ASK": 10,
        "LOG_LEVEL": "CRITICAL",
        "ASK_TIMEOUT_S": 5.0,
        "STREAM_TIMEOUT_S": 5.0,
        "STREAM_HEARTBEAT_S": 0.5,
        "CACHE_TTL_S": 600,
        "CACHE_MAX_ENTRIES": 500,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_result(
    n: int = 1,
    *,
    domain: str = "dsa",
    title: str = "Sliding Window",
    video_id: str = "vid1",
) -> SearchResult:
    """Build a SearchResult as retrieve.py would."""
    start = 60 * n
    text = f"chunk text {n}"
    return SearchResult(
        rank=n,
        score=0.9,
        chunk_id=f"c{n}",
        video_id=video_id,
        title=title,
        channel="Striver",
        domain=domain,
        start=start,
        end=start + 80,
        start_label=f"{start // 60:02d}:00",
        url=f"https://youtu.be/{video_id}?t={start}",
        text=text,
        snippet=text,
    )


def _filter_domain(qdrant_filter: Any) -> str | None:
    try:
        return qdrant_filter.must[0].match.value
    except (AttributeError, IndexError, TypeError):
        return None


class FakeClient:
    """Minimal Qdrant client supporting probe, count and scroll."""

    def __init__(
        self,
        *,
        points: int = 3,
        counts: dict[str, int] | None = None,
        video_payloads: list[dict[str, Any]] | None = None,
        up: bool = True,
    ) -> None:
        self.points = points
        self.up = up
        self.counts = counts or {"dsa": 3, "system_design": 1}
        self.video_payloads = video_payloads or [
            {"video_id": "v1", "title": "Sliding Window", "channel": "Striver", "domain": "dsa"},
            {"video_id": "v1", "title": "Sliding Window", "channel": "Striver", "domain": "dsa"},
            {"video_id": "v2", "title": "Binary Search", "channel": "Striver", "domain": "dsa"},
            {
                "video_id": "v3",
                "title": "Consistent Hashing",
                "channel": "Gaurav Sen",
                "domain": "system_design",
            },
        ]

    def get_collections(self) -> Any:
        if not self.up:
            raise ConnectionError("qdrant down")
        return SimpleNamespace(
            collections=[SimpleNamespace(name="youtube_reviser")]
        )

    def collection_exists(self, name: str) -> bool:
        return self.up

    def get_collection(self, name: str) -> Any:
        return SimpleNamespace(
            points_count=self.points,
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=384))
            ),
        )

    def count(
        self,
        collection_name: str,
        count_filter: Any = None,
        exact: bool = True,
    ) -> Any:
        return SimpleNamespace(count=self.counts.get(_filter_domain(count_filter), 0))

    def scroll(
        self,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 256,
        offset: Any = None,
        with_payload: Any = None,
        with_vectors: bool = False,
    ) -> tuple[list[Any], Any]:
        domain = _filter_domain(scroll_filter)
        rows = [
            payload
            for payload in self.video_payloads
            if domain is None or payload.get("domain") == domain
        ]
        start = int(offset or 0)
        page = rows[start : start + limit]
        next_offset = start + len(page) if start + len(page) < len(rows) else None
        return [SimpleNamespace(payload=payload) for payload in page], next_offset


class FakeRetriever:
    """Retriever double that never loads a model or touches the network."""

    def __init__(
        self,
        results: list[SearchResult] | None = None,
        *,
        down: bool = False,
        client: FakeClient | None = None,
    ) -> None:
        self.results = results if results is not None else [make_result(1)]
        self.down = down
        self._client = client or FakeClient()
        self.last_timings = {"embed_ms": 1.0, "search_ms": 2.0}
        self.last_results: list[SearchResult] = []
        self.search_calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def client(self) -> FakeClient:
        if self.down:
            raise RetrievalError("Qdrant is not reachable at http://localhost:6333.")
        return self._client

    def embed_query(self, query: str) -> list[float]:
        return [0.0] * 384

    def search(self, query: str, **kwargs: Any) -> list[SearchResult]:
        if self.down:
            raise RetrievalError("Qdrant is not reachable at http://localhost:6333.")
        self.search_calls.append((query, kwargs))
        self.last_results = list(self.results)
        return list(self.results)


class FakeLLM:
    """Scripted LLM: strings for normal calls, lists for streaming calls."""

    model = "fake/model"
    last_usage = {"input_tokens": 7, "output_tokens": 5}

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or ["Answer [1]"])
        self.calls: list[dict[str, Any]] = []

    def complete(self, system: str, user: str, stream: bool = False) -> Any:
        self.calls.append({"system": system, "user": user, "stream": stream})
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        item = self.responses.pop(0)
        if stream:
            return iter(item if isinstance(item, list) else [item])
        return item


@contextmanager
def api(
    *,
    results: list[SearchResult] | None = None,
    retriever: FakeRetriever | None = None,
    llm: FakeLLM | None = None,
    settings: Settings | None = None,
) -> Iterator[tuple[TestClient, FakeRetriever, FakeLLM]]:
    """Build an app with fakes and run its lifespan."""
    fake_retriever = retriever or FakeRetriever(results=results)
    fake_llm = llm or FakeLLM()
    app = create_app(
        settings or make_settings(),
        retriever=fake_retriever,
        llm=fake_llm,
    )
    with TestClient(app) as client:
        yield client, fake_retriever, fake_llm


@pytest.fixture(autouse=True)
def reset_limiter() -> Iterator[None]:
    """Keep slowapi's in-process counters isolated between tests."""
    limiter.reset()
    yield
    limiter.reset()


# --------------------------------------------------------------------------- #
# Health and search
# --------------------------------------------------------------------------- #
def test_health_ok_and_no_secrets() -> None:
    with api() as (client, _, _):
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["qdrant_reachable"] is True
    assert body["collection_points"] == 3
    assert body["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert body["llm_provider"] == "groq"
    assert body["llm_model"] == "fake/model"
    assert body["llm_ready"] is True
    assert response.headers["X-Request-ID"]
    assert "api_key" not in json.dumps(body).lower()


def test_search_happy_path() -> None:
    results = [make_result(1), make_result(2, title="Binary Search", video_id="vid2")]
    with api(results=results) as (client, retriever, _):
        response = client.post(
            "/search",
            json={
                "query": "sliding window",
                "domain": "dsa",
                "top_k": 5,
                "use_mmr": True,
                "use_reranker": False,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["query"] == "sliding window"
    assert body["results"][0]["chunk_id"] == "c1"
    assert body["latency_ms"] >= 0
    assert retriever.search_calls[-1][1]["domain"] == "dsa"
    assert retriever.search_calls[-1][1]["top_k"] == 5


def test_search_empty_results_is_200() -> None:
    with api(results=[]) as (client, _, _):
        response = client.post("/search", json={"query": "nothing matches here"})
    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["count"] == 0


# --------------------------------------------------------------------------- #
# Ask
# --------------------------------------------------------------------------- #
def test_ask_found_false_does_not_call_llm() -> None:
    llm = FakeLLM()
    with api(results=[], llm=llm) as (client, _, _):
        response = client.post("/ask", json={"query": "unknown topic"})
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is False
    assert body["sources"] == []
    assert body["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert llm.calls == []


def test_ask_cache_hit_on_repeated_query() -> None:
    llm = FakeLLM(["Cached answer [1]"])
    with api(results=[make_result(1)], llm=llm) as (client, _, _):
        first = client.post("/ask", json={"query": "sliding window", "domain": "dsa"})
        second = client.post("/ask", json={"query": "  sliding   window ", "domain": "dsa"})
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"
    assert first.json()["answer"] == second.json()["answer"]
    assert second.json()["query"] == "sliding   window"
    assert len(llm.calls) == 1


def test_ask_stream_token_then_done() -> None:
    llm = FakeLLM([["Sliding ", "window [1]"]])
    with api(results=[make_result(1)], llm=llm) as (client, _, _):
        with client.stream(
            "POST",
            "/ask/stream",
            json={"query": "explain sliding window", "domain": "dsa"},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            payload = "".join(response.iter_text())

    events: list[tuple[str, dict[str, Any]]] = []
    for block in payload.replace("\r\n", "\n").split("\n\n"):
        lines = [line for line in block.splitlines() if line]
        if not lines or lines[0].startswith(":"):
            continue
        event = next(
            (line[7:] for line in lines if line.startswith("event: ")), ""
        )
        data = next(
            (line[6:] for line in lines if line.startswith("data: ")), "{}"
        )
        events.append((event, json.loads(data)))

    names = [event for event, _ in events]
    assert names[0] == "token"
    assert names[-1] == "done"
    assert "token" in names
    tokens = "".join(
        data["text"] for event, data in events if event == "token"
    )
    assert tokens == "Sliding window [1]"
    done = next(data for event, data in events if event == "done")
    assert done["sources"][0]["n"] == 1
    assert done["found"] is True


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 301},
        {"query": "sliding window", "domain": "math"},
        {"query": "sliding window", "top_k": 0},
        {"query": "sliding window", "top_k": 21},
    ],
)
def test_search_validation_errors(body: dict[str, Any]) -> None:
    with api() as (client, _, _):
        response = client.post("/search", json=body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["request_id"] == response.headers["X-Request-ID"]


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def test_auth_disabled_without_api_key() -> None:
    with api() as (client, _, _):
        response = client.post("/search", json={"query": "sliding window"})
    assert response.status_code == 200


def test_auth_enabled_on_all_but_health() -> None:
    settings = make_settings(API_KEY="test-secret")
    with api(settings=settings) as (client, _, _):
        missing = client.post("/search", json={"query": "sliding window"})
        wrong = client.post(
            "/search",
            json={"query": "sliding window"},
            headers={"X-API-Key": "wrong"},
        )
        health = client.get("/health")
        good = client.post(
            "/search",
            json={"query": "sliding window"},
            headers={"X-API-Key": "test-secret"},
        )
    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "unauthorized"
    assert wrong.status_code == 401
    assert health.status_code == 200
    assert good.status_code == 200


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
def test_rate_limit_returns_429() -> None:
    settings = make_settings(RATE_LIMIT_SEARCH=1)
    with api(settings=settings) as (client, _, _):
        first = client.post("/search", json={"query": "sliding window"})
        second = client.post("/search", json={"query": "binary search"})
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limited"
    assert second.headers["Retry-After"] == "60"


# --------------------------------------------------------------------------- #
# Degraded / downstream failures
# --------------------------------------------------------------------------- #
def test_qdrant_down_degrades_health_and_returns_503() -> None:
    retriever = FakeRetriever(down=True)
    with api(retriever=retriever) as (client, _, _):
        health = client.get("/health")
        search = client.post("/search", json={"query": "sliding window"})
        ask = client.post("/ask", json={"query": "sliding window"})
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "degraded"
    assert body["qdrant_reachable"] is False
    assert body["collection_points"] == 0
    assert search.status_code == 503
    assert search.json()["error"]["code"] == "index_unavailable"
    assert ask.status_code == 503


def test_quiz_parse_failure_returns_502() -> None:
    llm = FakeLLM(["not json", "still not json"])
    with api(results=[make_result(1)], llm=llm) as (client, _, _):
        response = client.post(
            "/quiz",
            json={"topic": "binary search", "domain": "dsa", "num_questions": 3},
        )
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "quiz_parse_error"
    assert "Traceback" not in json.dumps(response.json())
    assert len(llm.calls) == 2


def test_quiz_success_returns_questions() -> None:
    payload = {
        "questions": [
            {
                "question": "What is binary search complexity?",
                "answer": "O(log n).",
                "source_n": 1,
            }
        ]
    }
    llm = FakeLLM([json.dumps(payload)])
    with api(results=[make_result(1)], llm=llm) as (client, _, _):
        response = client.post(
            "/quiz",
            json={"topic": "binary search", "domain": "dsa", "num_questions": 3},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["questions"][0]["source_n"] == 1
    assert body["sources"][0]["n"] == 1


def test_missing_llm_keeps_search_working() -> None:
    app = create_app(make_settings(), retriever=FakeRetriever(), llm=None)
    with TestClient(app) as client:
        health = client.get("/health")
        search = client.post("/search", json={"query": "sliding window"})
        ask = client.post("/ask", json={"query": "sliding window"})
    assert health.json()["status"] == "degraded"
    assert health.json()["llm_ready"] is False
    assert search.status_code == 200
    assert ask.status_code == 503
    assert ask.json()["error"]["code"] == "llm_unavailable"


def test_openapi_schema_and_docs_are_available() -> None:
    with api() as (client, _, _):
        schema = client.get("/openapi.json")
        docs = client.get("/docs")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    assert "/search" in paths
    assert "/ask/stream" in paths
    assert "/videos" in paths
    assert "X-API-Key" in json.dumps(schema.json())
    assert docs.status_code == 200


def test_revise_returns_note_and_sources() -> None:
    llm = FakeLLM(["Key idea: two pointers [1]."])
    with api(results=[make_result(1)], llm=llm) as (client, _, _):
        response = client.post(
            "/revise", json={"topic": "sliding window", "domain": "dsa"}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["note"] == "Key idea: two pointers [1]."
    assert body["sources"][0]["n"] == 1
    assert body["found"] is True


# --------------------------------------------------------------------------- #
# Index metadata
# --------------------------------------------------------------------------- #
def test_topics_and_videos_metadata() -> None:
    with api() as (client, _, _):
        topics = client.get("/topics")
        videos = client.get(
            "/videos", params={"domain": "dsa", "q": "sliding", "limit": 20, "offset": 0}
        )
    assert topics.status_code == 200
    topic_rows = {row["domain"]: row for row in topics.json()["topics"]}
    assert topic_rows["dsa"]["chunks"] == 3
    assert topic_rows["dsa"]["videos"] == 2
    assert topic_rows["system_design"]["chunks"] == 1
    assert topic_rows["system_design"]["videos"] == 1

    assert videos.status_code == 200
    body = videos.json()
    assert body["total"] == 1
    assert body["count"] == 1
    assert body["videos"][0]["video_id"] == "v1"
    assert body["videos"][0]["chunks"] == 2
