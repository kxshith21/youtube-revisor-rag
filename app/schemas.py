"""Pydantic v2 request/response models for the API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

# One domain per request (or None = all domains).
Domain = Literal["dsa", "system_design"]
# Generation modes exposed by scripts/generate.py.
Mode = Literal["answer", "revise", "quiz"]


def _strip(value: Any) -> Any:
    """Strip whitespace before length validation."""
    return value.strip() if isinstance(value, str) else value


CleanStr = Annotated[str, BeforeValidator(_strip)]

QUERY_FIELD = Field(
    ..., min_length=1, max_length=300, examples=["explain the sliding window pattern"]
)


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class SearchRequest(BaseModel):
    """Retrieval-only request (no LLM)."""

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "query": "what is sliding window",
            "domain": "dsa",
            "top_k": 5,
            "use_mmr": True,
            "use_reranker": False,
        }],
    })

    query: CleanStr = QUERY_FIELD
    domain: Domain | None = Field(
        None, description="Restrict to one domain, or null to search everything.",
        examples=["dsa"],
    )
    top_k: int = Field(
        5, ge=1, le=20, description="Number of results (1-20).", examples=[5],
    )
    use_mmr: bool = Field(
        True, description="Diversify the results (MMR).", examples=[True],
    )
    use_reranker: bool = Field(
        False, description="Cross-encoder rerank (slower, more accurate).",
        examples=[False],
    )


class HistoryItem(BaseModel):
    """One previous question/answer pair (max 3)."""

    question: CleanStr = Field(
        ..., min_length=1, max_length=500,
        examples=["explain sliding window"],
    )
    answer: str = Field(
        ..., max_length=4000, examples=["A window is a contiguous range [1]..."],
    )


class AskRequest(BaseModel):
    """Full RAG answer request."""

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "query": "what about its time complexity?",
            "domain": "dsa",
            "top_k": 5,
            "history": [{
                "question": "explain sliding window",
                "answer": "A contiguous range of fixed or variable size [1].",
            }],
        }],
    })

    query: CleanStr = QUERY_FIELD
    domain: Domain | None = Field(None, examples=["dsa"])
    top_k: int = Field(5, ge=1, le=20, examples=[5])
    history: list[HistoryItem] = Field(
        default_factory=list, max_length=3,
        description="Last question/answer pairs, used to rewrite follow-ups.",
    )


class ReviseRequest(BaseModel):
    """Revision-note request (mode=revise)."""

    model_config = ConfigDict(json_schema_extra={
        "examples": [{"topic": "binary search", "domain": "dsa"}],
    })

    topic: CleanStr = QUERY_FIELD
    domain: Domain | None = Field(None, examples=["dsa"])


class QuizRequest(BaseModel):
    """Quiz request (mode=quiz)."""

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "topic": "consistent hashing",
            "domain": "system_design",
            "num_questions": 5,
        }],
    })

    topic: CleanStr = QUERY_FIELD
    domain: Domain | None = Field(None, examples=["system_design"])
    num_questions: int = Field(
        5, ge=3, le=10, description="Questions to generate (3-10).", examples=[5],
    )


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class SearchResultOut(BaseModel):
    """One retrieved chunk with a clickable timestamp."""

    rank: int
    score: float
    chunk_id: str
    video_id: str
    title: str
    channel: str
    domain: str
    start: int
    end: int
    start_label: str
    url: str
    text: str
    snippet: str


class SourceOut(BaseModel):
    """A cited source, built from retrieval results in code (never from the LLM)."""

    n: int = Field(..., description="Citation number used in the answer, e.g. [1].")
    title: str
    start_label: str
    url: str
    domain: str
    score: float


class AnswerOut(BaseModel):
    """A grounded answer with validated citations."""

    query: str
    answer: str
    sources: list[SourceOut]
    found: bool = Field(..., description="False when nothing relevant was found.")
    uncited: bool = Field(..., description="True when the answer cites no sources.")
    model: str
    latency_ms: dict[str, float]
    usage: dict[str, int]


class SearchResponse(BaseModel):
    """Result of POST /search."""

    query: str
    results: list[SearchResultOut]
    count: int
    latency_ms: float


class ReviseResponse(BaseModel):
    """Revision note returned by POST /revise."""

    topic: str
    note: str
    sources: list[SourceOut]
    found: bool = Field(..., description="False when nothing relevant was found.")
    uncited: bool = Field(..., description="True when the note cites no sources.")
    model: str
    latency_ms: dict[str, float]
    usage: dict[str, int]


class QuizQuestion(BaseModel):
    """One generated quiz question (cited by block number)."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(..., examples=["What is the time complexity of binary search?"])
    answer: str = Field(..., examples=["O(log n), because the search space halves each step."])
    source_n: int = Field(..., description="Context block number the question came from.")


class QuizResponse(BaseModel):
    """Result of POST /quiz."""

    questions: list[QuizQuestion]
    sources: list[SourceOut]
    found: bool = Field(
        True, description="False when no relevant indexed content was found.",
    )
    model: str
    latency_ms: dict[str, float]
    usage: dict[str, int]


class HealthOut(BaseModel):
    """Service health (no secrets)."""

    status: Literal["ok", "degraded"]
    qdrant_reachable: bool
    collection_points: int
    embedding_model: str
    llm_provider: str
    llm_model: str
    llm_ready: bool
    detail: str | None = Field(None, description="Why the service is degraded, if it is.")


class TopicOut(BaseModel):
    """Per-domain counts."""

    domain: str
    videos: int
    chunks: int


class TopicsResponse(BaseModel):
    """Result of GET /topics."""

    topics: list[TopicOut]
    count: int


class VideoOut(BaseModel):
    """One distinct indexed video."""

    video_id: str
    title: str
    channel: str
    domain: str
    chunks: int


class VideosResponse(BaseModel):
    """Result of GET /videos."""

    videos: list[VideoOut]
    count: int
    total: int
    limit: int
    offset: int


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ErrorDetail(BaseModel):
    """Machine-readable error payload."""

    code: str = Field(..., description="Stable error code, e.g. 'validation_error'.")
    message: str
    request_id: str


class ErrorEnvelope(BaseModel):
    """Every non-2xx response has this shape."""

    error: ErrorDetail
