"""Run the YouTube Reviser RAG pipeline against the golden dataset.

The existing pipeline is imported, never modified:

    scripts/retrieve.py    Retriever, SearchResult, build_context, cap_context
    scripts/generate.py    generate, Answer

Two transports are supported:

    (default)   direct in-process calls into scripts/generate.py
    --via-api   POST /ask + POST /search against a running FastAPI server

Every question is evaluated independently. One failure is recorded and the
run continues. Results are cached per question + config hash so ``--resume``
(and repeated runs) do not pay for the same work twice.

Usage:
    python eval/run_eval.py --split dev --top-k 5 --yes
    python eval/run_eval.py --split test --judge-provider groq --judge-model openai/gpt-oss-120b
    python eval/run_eval.py --via-api --config-name api-baseline --yes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

# Load the project .env before importing config-backed modules.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:  # pragma: no cover - project dependency but stay safe
    pass

import metrics  # noqa: E402
import report as report_module  # noqa: E402
from build_golden import (  # noqa: E402
    CHUNKS_DIR,
    DEFAULT_SEED,
    DATA_DIR,
    GoldenQuestion,
    load_golden,
)
from config import LLM_MODEL, LLM_PROVIDER  # noqa: E402
from judge import Judge, JudgeResult  # noqa: E402
from llm import (  # noqa: E402
    PROMPT_VERSION,
    LLMConfigError,
    estimate_call_tokens,
    generator_spec,
    resolve_judge_defaults,
)

try:  # Optional dependency used only by run/report tooling.
    import httpx
except ImportError:  # pragma: no cover - httpx is in requirements.txt
    httpx = None  # type: ignore[assignment]

RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = EVAL_DIR / ".cache"
DEFAULT_API_BASE = os.getenv("EVAL_API_BASE", "http://localhost:8000")
DEFAULT_API_PREFIX = os.getenv("EVAL_API_PREFIX", "/api/v1")
CONTEXT_MAX_CHARS = 6000
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    """Everything that can change a measured number."""

    top_k: int = 5
    threshold: float = 0.35
    use_mmr: bool = True
    use_reranker: bool = False
    min_overlap: float = 10.0
    domain: str | None = None
    split: str = "all"
    limit: int | None = None
    seed: int = DEFAULT_SEED
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_retrieval: bool = False
    no_judge: bool = False
    via_api: bool = False
    api_base: str = DEFAULT_API_BASE
    api_prefix: str = DEFAULT_API_PREFIX
    sleep_seconds: float = 2.0
    resume: bool = False
    config_name: str = ""
    golden_path: Path = DATA_DIR / "golden.jsonl"
    results_dir: Path = RESULTS_DIR
    cache_dir: Path = CACHE_DIR
    chunks_dir: Path = CHUNKS_DIR

    def hash_payload(self, index_snapshot: dict[str, Any]) -> dict[str, Any]:
        """Only knobs that can change measurements belong here."""
        return {
            "prompt_version": PROMPT_VERSION,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "use_mmr": self.use_mmr,
            "use_reranker": self.use_reranker,
            "min_overlap": self.min_overlap,
            "domain": self.domain,
            "split": self.split,
            "via_api": self.via_api,
            "generator": generator_spec(),
            "judge_provider": None if self.no_judge else self.judge_provider,
            "judge_model": None if self.no_judge else self.judge_model,
            "judge_retrieval": False if self.no_judge else self.judge_retrieval,
            "index_snapshot": index_snapshot,
        }

    def config_hash(self, index_snapshot: dict[str, Any]) -> str:
        blob = json.dumps(self.hash_payload(index_snapshot), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def public_dict(self) -> dict[str, Any]:
        """Config safe to print/store: no secrets, no absolute temp paths."""
        data = asdict(self)
        for key in ("golden_path", "results_dir", "cache_dir", "chunks_dir"):
            data[key] = str(data[key])
        return data


# --------------------------------------------------------------------------- #
# Index metadata / gold chunk text
# --------------------------------------------------------------------------- #
def compute_index_snapshot(chunks_dir: Path = CHUNKS_DIR) -> dict[str, Any]:
    """Cheap coverage fingerprint from the chunk files."""
    snapshot: dict[str, Any] = {}
    for domain in ("dsa", "system_design"):
        path = chunks_dir / f"{domain}.jsonl"
        if not path.exists():
            continue
        videos: set[str] = set()
        chunks = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                videos.add(str(row.get("video_id", "")))
                chunks += 1
        snapshot[f"{domain}_videos"] = len(videos)
        snapshot[f"{domain}_chunks"] = chunks
    snapshot["total_chunks"] = sum(
        value for key, value in snapshot.items() if key.endswith("_chunks")
    )
    return snapshot


def load_chunk_lookup(chunks_dir: Path = CHUNKS_DIR) -> dict[str, list[dict[str, Any]]]:
    """video_id -> [{start,end,text}] used for retrieval-judge gold excerpts."""
    lookup: dict[str, list[dict[str, Any]]] = {}
    for domain in ("dsa", "system_design"):
        path = chunks_dir / f"{domain}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lookup.setdefault(str(row.get("video_id", "")), []).append(
                    {
                        "start": float(row.get("start", 0) or 0),
                        "end": float(row.get("end", 0) or 0),
                        "text": str(row.get("text", "")),
                    }
                )
    return lookup


def gold_excerpts(
    question: GoldenQuestion,
    lookup: dict[str, list[dict[str, Any]]],
    *,
    limit: int = 3,
) -> list[str]:
    """Transcript text of chunks overlapping the gold spans."""
    excerpts: list[str] = []
    for source in question.gold_sources:
        for chunk in lookup.get(source.video_id, []):
            if metrics.overlap_seconds(
                chunk["start"], chunk["end"], source.start, source.end
            ) <= 0:
                continue
            text = chunk["text"].strip()
            if text:
                excerpts.append(text[:800])
            if len(excerpts) >= limit:
                return excerpts
    return excerpts


def validate_gold_videos(
    questions: Sequence[GoldenQuestion], chunks_dir: Path = CHUNKS_DIR
) -> None:
    """Fail loudly when a gold video id is not in the index files."""
    known: set[str] = set()
    found_any = False
    for domain in ("dsa", "system_design"):
        path = chunks_dir / f"{domain}.jsonl"
        if not path.exists():
            continue
        found_any = True
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    known.add(str(json.loads(line).get("video_id", "")))
                except json.JSONDecodeError:
                    continue
    if not found_any:
        print(
            "  WARNING: no chunk files found; skipping gold video id validation",
            file=sys.stderr,
        )
        return
    missing = sorted(
        {
            f"{question.id}:{source.video_id}"
            for question in questions
            for source in question.gold_sources
            if source.video_id not in known
        }
    )
    if missing:
        raise ValueError(
            "Gold video_id(s) are not in the index: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )


# --------------------------------------------------------------------------- #
# Result item helpers (works for SearchResult and API dicts)
# --------------------------------------------------------------------------- #
def item_get(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def format_context(items: Sequence[Any]) -> str:
    blocks = [
        f"[{index}] {item_get(item, 'title')} @ {item_get(item, 'start_label')} "
        f"({item_get(item, 'url')})\n{item_get(item, 'text')}"
        for index, item in enumerate(items, start=1)
    ]
    return "\n\n".join(blocks)


def cap_items(items: Sequence[Any], max_chars: int = CONTEXT_MAX_CHARS) -> list[Any]:
    """Mirror scripts.generate.cap_context without importing SearchResult."""
    kept = list(items)
    while kept and len(format_context(kept)) > max_chars:
        kept.pop()
    return kept


def item_to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {
        key: getattr(item, key, None)
        for key in (
            "rank",
            "score",
            "chunk_id",
            "video_id",
            "title",
            "channel",
            "domain",
            "start",
            "end",
            "start_label",
            "url",
            "text",
            "snippet",
        )
    }


def spans_from_items(items: Sequence[Any]) -> list[metrics.Span]:
    return [
        metrics.Span(
            video_id=str(item_get(item, "video_id", "")),
            start=float(item_get(item, "start", 0) or 0),
            end=float(item_get(item, "end", 0) or 0),
        )
        for item in items
    ]


def build_citation_items(
    answer_text: str, sources: Sequence[dict[str, Any]], kept: Sequence[Any]
) -> list[dict[str, Any]]:
    """One citation record per actually cited block, with its sentence."""
    sentences = re.split(r"(?<=[.!?])\s+", answer_text or "")
    cited_ns = sorted(
        {
            int(source.get("n", 0))
            for source in sources
            if str(source.get("n", "")).isdigit()
        }
    )
    items: list[dict[str, Any]] = []
    for n in cited_ns:
        if not 1 <= n <= len(kept):
            continue
        marker = f"[{n}]"
        sentence = next((s for s in sentences if marker in s), "")
        if not sentence:
            continue
        source = kept[n - 1]
        items.append(
            {
                "n": n,
                "sentence": sentence.strip()[:800],
                "source_title": str(item_get(source, "title", "")),
                "source_start_label": str(item_get(source, "start_label", "")),
                "source_text": str(item_get(source, "text", ""))[:2000],
            }
        )
    return items


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
@dataclass
class PipelineOutput:
    answer: str
    sources: list[dict[str, Any]]
    found: bool
    uncited: bool
    model: str
    usage: dict[str, int]
    retrieved: list[Any]
    context: str
    latency_ms: dict[str, float]


class ScopedRetriever:
    """Inject eval settings into the existing Retriever.search without editing it."""

    def __init__(
        self, base: Any, domain: str | None, config: RunConfig, capture: dict[str, Any]
    ) -> None:
        self.base = base
        self.domain = domain
        self.config = config
        self.capture = capture

    def search(self, query: str, **kwargs: Any) -> Any:
        kwargs["domain"] = self.domain
        kwargs["score_threshold"] = self.config.threshold
        kwargs["use_mmr"] = self.config.use_mmr
        kwargs["use_reranker"] = self.config.use_reranker
        results = self.base.search(query, **kwargs)
        self.capture["results"] = list(results)
        return results

    @property
    def last_timings(self) -> dict[str, float]:
        return getattr(
            self.base, "last_timings", {"embed_ms": 0.0, "search_ms": 0.0}
        )


class DirectPipeline:
    """Call scripts/generate.generate in-process with eval overrides."""

    def __init__(
        self,
        config: RunConfig,
        retriever: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        self.config = config
        self._retriever = retriever
        self._llm = llm

    def _ensure(self) -> None:
        if self._retriever is None:
            from retrieve import Retriever

            self._retriever = Retriever()
        if self._llm is None:
            from generate import make_llm

            self._llm = make_llm()

    def run(self, question: GoldenQuestion) -> PipelineOutput:
        from generate import generate

        self._ensure()
        capture: dict[str, Any] = {"results": [], "context": ""}
        domain = None if question.category == "off_topic" else question.domain
        scoped = ScopedRetriever(self._retriever, domain, self.config, capture)
        started = time.perf_counter()
        answer = generate(
            query=question.question,
            retriever=scoped,
            llm=self._llm,
            top_k=self.config.top_k,
            mode="answer",
            on_context=lambda context: capture.__setitem__("context", context),
        )
        wall_ms = round((time.perf_counter() - started) * 1000)
        latency = {
            "retrieval": float(answer.latency_ms.get("retrieval", 0) or 0),
            "generation": float(answer.latency_ms.get("generation", 0) or 0),
            "total": float(wall_ms),
        }
        return PipelineOutput(
            answer=str(answer.answer or ""),
            sources=list(answer.sources or []),
            found=bool(answer.found),
            uncited=bool(answer.uncited),
            model=str(answer.model or ""),
            usage={key: int(value or 0) for key, value in (answer.usage or {}).items()},
            retrieved=list(capture["results"]),
            context=str(capture["context"] or ""),
            latency_ms=latency,
        )


@dataclass
class ApiAnswer:
    """AnswerOut-shaped object so both transports share downstream code."""

    query: str
    answer: str
    sources: list[dict[str, Any]]
    found: bool
    uncited: bool
    model: str
    latency_ms: dict[str, float]
    usage: dict[str, int]


class ApiPipeline:
    """Use a running FastAPI server. Retrieval comes from POST /search."""

    def __init__(self, config: RunConfig) -> None:
        if httpx is None:
            raise RuntimeError("--via-api requires httpx (pip install httpx).")
        self.config = config
        self._client = httpx.Client(
            base_url=config.api_base.rstrip("/"),
            timeout=180.0,
        )
        self._prefix = self._resolve_prefix()
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = os.getenv("API_KEY")
        if api_key:
            self._headers["X-API-Key"] = api_key

    def _resolve_prefix(self) -> str:
        candidates = [self.config.api_prefix]
        if self.config.api_prefix:
            candidates.append("")  # current app exposes root-level routes
        for prefix in dict.fromkeys(candidates):
            try:
                response = self._client.get(f"{prefix}/health")
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
            if response.status_code < 500:
                if prefix != self.config.api_prefix:
                    print(
                        f"  API prefix fallback: using {prefix!r} "
                        f"(requested {self.config.api_prefix!r})"
                    )
                return prefix
        raise RuntimeError(
            f"Cannot reach the API at {self.config.api_base}. "
            "Start it with: uvicorn app.main:app --port 8000"
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._prefix}{path}"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.post(url, json=payload, headers=self._headers)
                if response.status_code in RETRYABLE_STATUS:
                    last_error = RuntimeError(
                        f"{path} returned {response.status_code}"
                    )
                elif response.status_code >= 400:
                    raise RuntimeError(
                        f"{path} returned {response.status_code}: "
                        f"{response.text[:300]}"
                    )
                else:
                    return response.json()
            except Exception as exc:  # noqa: BLE001 - classify below
                if isinstance(exc, RuntimeError) and "returned 4" in str(exc):
                    raise
                last_error = exc
            if attempt < 2:
                wait = 2.0 * (2**attempt)
                print(
                    f"  API retry {attempt + 1}/2 in {wait:.0f}s "
                    f"({type(last_error).__name__})",
                    file=sys.stderr,
                )
                time.sleep(wait)
        raise RuntimeError(f"{path} failed after retries: {last_error}")

    def run(self, question: GoldenQuestion) -> PipelineOutput:
        domain = None if question.category == "off_topic" else question.domain
        started = time.perf_counter()
        answer_payload = self._post(
            "/ask",
            {
                "query": question.question,
                "domain": domain,
                "top_k": self.config.top_k,
            },
        )
        search_payload = self._post(
            "/search",
            {
                "query": question.question,
                "domain": domain,
                "top_k": self.config.top_k,
                "use_mmr": self.config.use_mmr,
                "use_reranker": self.config.use_reranker,
            },
        )
        wall_ms = round((time.perf_counter() - started) * 1000)
        retrieved = list(search_payload.get("results") or [])
        kept = cap_items(retrieved)
        context = format_context(kept)
        ask_latency = answer_payload.get("latency_ms") or {}
        retrieval_ms = float(ask_latency.get("retrieval", 0) or 0) + float(
            search_payload.get("latency_ms", 0) or 0
        )
        generation_ms = float(ask_latency.get("generation", 0) or 0)
        return PipelineOutput(
            answer=str(answer_payload.get("answer", "") or ""),
            sources=list(answer_payload.get("sources") or []),
            found=bool(answer_payload.get("found", False)),
            uncited=bool(answer_payload.get("uncited", False)),
            model=str(answer_payload.get("model", "") or ""),
            usage={
                key: int(value or 0)
                for key, value in (answer_payload.get("usage") or {}).items()
            },
            retrieved=retrieved,
            context=context,
            latency_ms={
                "retrieval": retrieval_ms,
                "generation": generation_ms,
                "total": float(wall_ms),
            },
        )

    def close(self) -> None:
        self._client.close()


# --------------------------------------------------------------------------- #
# Evaluation of one question
# --------------------------------------------------------------------------- #
def _judge_payload(result: JudgeResult) -> dict[str, Any]:
    return {
        "value": result.value,
        "reason": result.reason,
        "details": result.details,
        "error": result.error,
    }


def _judge_usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        "input_tokens": int(after.get("input_tokens", 0))
        - int(before.get("input_tokens", 0)),
        "output_tokens": int(after.get("output_tokens", 0))
        - int(before.get("output_tokens", 0)),
    }


def evaluate_one(
    question: GoldenQuestion,
    pipeline: Any,
    config: RunConfig,
    judge: Judge | None,
    chunk_lookup: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run one question, compute local retrieval metrics, and (optionally) judge it.

    Never raises: transport, pipeline, and judge failures become data.
    """
    record: dict[str, Any] = {
        "id": question.id,
        "question": question.question,
        "domain": question.domain,
        "category": question.category,
        "answerable": question.answerable,
        "split": question.split,
        "gold_sources": [source.model_dump() for source in question.gold_sources],
        "reference_answer": question.reference_answer,
        "key_points": list(question.key_points),
        "status": "ok",
        "error": None,
        "answer": "",
        "found": False,
        "abstained": False,
        "uncited": False,
        "model": "",
        "retrieved": [],
        "context": "",
        "sources": [],
        "latency_ms": {},
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "judge_usage": {"input_tokens": 0, "output_tokens": 0},
        "retrieval": None,
        "judge": {},
    }

    judge_before = dict(judge.total_usage) if judge is not None else {}
    try:
        output = pipeline.run(question)
    except Exception as exc:  # noqa: BLE001 - a bad question must not stop the run
        record.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6),
            }
        )
        print(f"  ERROR {question.id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return record

    retrieved = list(output.retrieved)
    answer_text = output.answer or ""
    abstained = metrics.is_abstention(answer_text, output.found)
    record.update(
        {
            "answer": answer_text,
            "found": output.found,
            "abstained": abstained,
            "uncited": output.uncited,
            "model": output.model,
            "retrieved": [item_to_dict(item) for item in retrieved],
            "context": output.context,
            "sources": output.sources,
            "latency_ms": output.latency_ms,
            "usage": {
                "input_tokens": int(output.usage.get("input_tokens", 0) or 0),
                "output_tokens": int(output.usage.get("output_tokens", 0) or 0),
            },
        }
    )

    if question.answerable:
        record["retrieval"] = metrics.retrieval_metrics(
            spans_from_items(retrieved),
            [metrics.span_from_dict(source.model_dump()) for source in question.gold_sources],
            k=config.top_k,
            min_overlap=config.min_overlap,
        )
    else:
        record["retrieval"] = None

    if judge is None:
        return record

    judge_results: dict[str, JudgeResult] = {}
    if question.answerable and not abstained and answer_text.strip():
        judge_results["faithfulness"] = judge.faithfulness(
            question.question, answer_text, output.context
        )
        judge_results["answer_relevance"] = judge.answer_relevance(
            question.question, answer_text
        )
        if question.reference_answer.strip() and question.key_points:
            judge_results["correctness"] = judge.correctness(
                question.question,
                answer_text,
                question.reference_answer,
                question.key_points,
            )
        kept = cap_items(retrieved)
        citations = build_citation_items(answer_text, output.sources, kept)
        if citations:
            judge_results["citation_accuracy"] = judge.citation_accuracy(
                question.question, answer_text, citations
            )
        else:
            judge_results["citation_accuracy"] = JudgeResult(
                value=None, details={"skipped": "no citations"}
            )
        if config.judge_retrieval and record["retrieval"] is not None:
            gold_spans = [
                metrics.span_from_dict(source.model_dump())
                for source in question.gold_sources
            ]
            spans = spans_from_items(retrieved)
            matched = [
                any(
                    metrics.is_relevant(span, gold, config.min_overlap)
                    for gold in gold_spans
                )
                for span in spans
            ]
            excerpts = gold_excerpts(question, chunk_lookup or {})
            judged: list[bool | None] = []
            verdicts: list[dict[str, Any]] = []
            for index, (span, is_match) in enumerate(zip(spans, matched)):
                if is_match:
                    judged.append(True)
                    continue
                item = retrieved[index]
                result = judge.retrieval_relevance(
                    question.question,
                    question.reference_answer,
                    question.key_points,
                    excerpts,
                    str(item_get(item, "title", "")),
                    str(item_get(item, "text", ""))[:2000],
                )
                verdicts.append(
                    {
                        "index": index,
                        "video_id": span.video_id,
                        "start": span.start,
                        "relevant": result.value,
                        "reason": result.reason,
                        "error": result.error,
                    }
                )
                judged.append(bool(result.value) if result.value is not None else False)
            record["retrieval"]["judged_precision"] = metrics.judged_precision(
                matched, judged
            )
            record["retrieval"]["judge_verdicts"] = verdicts

    record["judge"] = {
        name: _judge_payload(result) for name, result in judge_results.items()
    }
    record["judge_usage"] = _judge_usage_delta(judge_before, judge.total_usage)
    return record


# --------------------------------------------------------------------------- #
# Cache / selection / cost
# --------------------------------------------------------------------------- #
def cache_path(cache_dir: Path, config_hash: str, question_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", question_id)
    return cache_dir / "runs" / config_hash / f"{safe}.json"


def load_cached(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if data.get("status") == "ok" else None


def save_cached(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        print(f"  WARNING: cache write failed ({exc})", file=sys.stderr)


def select_questions(
    questions: Sequence[GoldenQuestion], config: RunConfig
) -> list[GoldenQuestion]:
    selected = list(questions)
    if config.split != "all":
        selected = [question for question in selected if question.split == config.split]
    if config.domain:
        selected = [
            question
            for question in selected
            if question.answerable and question.domain == config.domain
        ]
    if config.limit is not None:
        selected = selected[: max(0, config.limit)]
    return selected


def estimate_run(
    questions: Sequence[GoldenQuestion], config: RunConfig
) -> dict[str, Any]:
    """Rough call/token estimate used for the pre-run confirmation."""
    n = len(questions)
    answerable = sum(1 for question in questions if question.answerable)
    generator_calls = n
    judge_calls = 0
    for question in questions:
        if not question.answerable:
            continue
        judge_calls += 2  # faithfulness + relevance
        if question.key_points and question.reference_answer:
            judge_calls += 1
        judge_calls += 1  # citation accuracy (skipped only when uncited)
    if config.judge_retrieval:
        judge_calls += answerable * config.top_k
    if config.no_judge:
        judge_calls = 0
    generator_tokens = generator_calls * estimate_call_tokens(
        "x" * 9000, output_tokens=500
    )
    judge_tokens = judge_calls * estimate_call_tokens("x" * 3500, output_tokens=350)
    return {
        "questions": n,
        "answerable": answerable,
        "generator_calls": generator_calls,
        "judge_calls": judge_calls,
        "estimated_tokens": generator_tokens + judge_tokens,
        "estimated_generator_tokens": generator_tokens,
        "estimated_judge_tokens": judge_tokens,
    }


def print_estimate(estimate: dict[str, Any], config: RunConfig) -> None:
    print(
        f"Run plan: {estimate['questions']} questions "
        f"({estimate['answerable']} answerable), ~"
        f"{estimate['generator_calls']} generation calls, "
        f"~{estimate['judge_calls']} judge calls, "
        f"~{estimate['estimated_tokens']:,} tokens."
    )
    print(
        f"  retrieval: top_k={config.top_k}, threshold={config.threshold}, "
        f"mmr={config.use_mmr}, rerank={config.use_reranker}, "
        f"min_overlap={config.min_overlap}s | sleep={config.sleep_seconds}s"
    )
    print("  Token counts are estimates; provider pricing is billed separately.")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def make_run_dir(config: RunConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", config.config_name).strip("-")
    folder = f"{stamp}-{name}" if name else stamp
    path = config.results_dir / folder
    suffix = 1
    while path.exists():
        path = config.results_dir / f"{folder}-{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def run(
    config: RunConfig,
    *,
    pipeline: Any | None = None,
    judge: Judge | None = None,
    questions: Sequence[GoldenQuestion] | None = None,
    out_dir: Path | None = None,
    confirm: bool = True,
) -> dict[str, Any]:
    """Run the whole evaluation and write reports. Returns the summary dict."""
    golden_path = config.golden_path
    questions = list(questions) if questions is not None else load_golden(golden_path)
    validate_gold_videos(questions, config.chunks_dir)
    selected = select_questions(questions, config)
    if not selected:
        raise ValueError(
            "No questions selected. Check --split/--domain/--limit and golden.jsonl."
        )

    snapshot = compute_index_snapshot(config.chunks_dir)
    if not snapshot:
        snapshot = selected[0].index_snapshot if selected else {}
    config_hash = config.config_hash(snapshot)

    if judge is None and not config.no_judge:
        provider, model = resolve_judge_defaults(
            config.judge_provider, config.judge_model
        )
        config.judge_provider, config.judge_model = provider, model
        config_hash = config.config_hash(snapshot)
        judge = Judge(
            provider=provider,
            model=model,
            sleep_seconds=config.sleep_seconds,
        )
    elif judge is not None:
        config.judge_provider, config.judge_model = judge.provider, judge.model
        config_hash = config.config_hash(snapshot)

    estimate = estimate_run(selected, config)
    print_estimate(estimate, config)
    if confirm and not config.no_judge and config.judge_provider:
        print(f"  judge: {config.judge_provider}/{config.judge_model}")
    if config.via_api and abs(config.threshold - 0.35) > 1e-9:
        print(
            "  NOTE: --via-api cannot override the server's retrieval threshold; "
            "--threshold only affects the local /search call if the server allows it.",
            file=sys.stderr,
        )
    if confirm:
        try:
            answer = input("Proceed with the full run? [y/N] ").strip().casefold()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Aborted. Re-run with --yes to skip this prompt.")
            raise SystemExit(1)

    out_dir = out_dir or make_run_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "config": config.public_dict(),
                "config_hash": config_hash,
                "index_snapshot": snapshot,
                "generator": generator_spec(),
                "judge": None
                if config.no_judge
                else {"provider": config.judge_provider, "model": config.judge_model},
                "estimate": estimate,
                "created_at": created_at,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    chunk_lookup = load_chunk_lookup(config.chunks_dir) if config.judge_retrieval else {}
    own_pipeline = pipeline is None
    if pipeline is None:
        pipeline = ApiPipeline(config) if config.via_api else DirectPipeline(config)

    records: list[dict[str, Any]] = []
    journal_path = out_dir / "questions.jsonl"
    print(f"\nRunning {len(selected)} questions -> {out_dir}")
    try:
        for index, question in enumerate(selected, start=1):
            path = cache_path(config.cache_dir, config_hash, question.id)
            cached = load_cached(path) if config.resume else None
            if cached is not None:
                records.append(cached)
                print(f"  [{index}/{len(selected)}] {question.id} (cached)")
            else:
                record = evaluate_one(
                    question, pipeline, config, judge, chunk_lookup
                )
                records.append(record)
                save_cached(path, record)
                status = record["status"]
                extra = ""
                if status == "ok":
                    extra = " abstained" if record["abstained"] else ""
                elif record.get("error"):
                    extra = f" {record['error'][:80]}"
                print(f"  [{index}/{len(selected)}] {question.id} {status}{extra}")
                if config.sleep_seconds > 0 and index < len(selected):
                    time.sleep(config.sleep_seconds)
            # Journal after every question so an interruption loses at most one.
            with journal_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(records[-1], ensure_ascii=False) + "\n"
                )
    finally:
        if own_pipeline and config.via_api and hasattr(pipeline, "close"):
            pipeline.close()

    summary = report_module.write_reports(
        records,
        {
            "config": config.public_dict(),
            "config_hash": config_hash,
            "index_snapshot": snapshot,
            "generator": generator_spec(),
            "judge": None
            if config.no_judge
            else {"provider": config.judge_provider, "model": config.judge_model},
            "estimate": estimate,
            "created_at": created_at,
        },
        out_dir,
    )
    if judge is not None:
        summary["judge_usage"] = dict(judge.total_usage)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"\nWrote {out_dir / 'summary.md'}")
    print(f"Wrote {out_dir / 'failures.md'}")
    print(f"Wrote {out_dir / 'summary.json'}")
    print()
    print(report_module.render_console(summary))
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the YouTube Reviser RAG pipeline on golden.jsonl."
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--no-mmr", action="store_true", help="disable MMR diversity")
    parser.add_argument("--rerank", action="store_true", help="enable the cross-encoder")
    parser.add_argument(
        "--domain", choices=["dsa", "system_design"], default=None,
        help="evaluate only this domain (off-topic probes are excluded)",
    )
    parser.add_argument(
        "--split", choices=["dev", "test", "all"], default="all",
        help="which golden split to run (default: all)",
    )
    parser.add_argument("--limit", type=int, default=None, help="first N questions")
    parser.add_argument("--min-overlap", type=float, default=10.0)
    parser.add_argument("--config-name", default="", help="label for the results folder")
    parser.add_argument("--judge-provider", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--judge-retrieval", action="store_true",
        help="also ask the judge about non-gold retrieved chunks (slow)",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="retrieval/abstention metrics only (no generation judge)",
    )
    parser.add_argument("--via-api", action="store_true")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-prefix", default=DEFAULT_API_PREFIX)
    parser.add_argument(
        "--sleep-between-calls", type=float, default=2.0,
        help="seconds between generation/judge calls (default: 2)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse cached successful answers for this config hash",
    )
    parser.add_argument("--golden", type=Path, default=DATA_DIR / "golden.jsonl")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--yes", action="store_true", help="skip the cost prompt")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        top_k=args.top_k,
        threshold=args.threshold,
        use_mmr=not args.no_mmr,
        use_reranker=args.rerank,
        min_overlap=args.min_overlap,
        domain=args.domain,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_retrieval=args.judge_retrieval,
        no_judge=args.no_judge,
        via_api=args.via_api,
        api_base=args.api_base,
        api_prefix=args.api_prefix,
        sleep_seconds=max(0.0, args.sleep_between_calls),
        resume=args.resume,
        config_name=args.config_name,
        golden_path=args.golden,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        chunks_dir=args.chunks_dir,
    )


def main(argv: list[str] | None = None) -> None:
    report_module.enable_utf8_console()
    args = parse_args(argv)
    config = config_from_args(args)
    try:
        run(config, confirm=not args.yes)
    except (FileNotFoundError, ValueError, LLMConfigError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
