"""Retrieval module for the YouTube Reviser RAG project.

Takes a question, embeds it, searches Qdrant, post-processes the hits
(threshold -> overlap merge -> MMR -> optional rerank), and returns
UI-ready SearchResult dicts. No LLM calls in this file.

Usage (from the project root):
    python scripts/retrieve.py "what is sliding window" --domain dsa --top-k 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

# Shared constants (MODEL_NAME, QUERY_PREFIX, QDRANT_URL, ...) - never duplicated.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    COLLECTION_NAME,
    MODEL_NAME,
    QDRANT_API_KEY,
    QDRANT_DOCKER_CMD,
    QDRANT_URL,
    QUERY_PREFIX,
    SEARCH_HNSW_EF,
    VECTOR_SIZE,
)
from chunk_transcripts import format_label, load_glossary  # noqa: E402
from embed_and_store import get_device  # noqa: E402

MAX_QUERY_CHARS = 300
DEFAULT_TOP_K = 5
DEFAULT_FETCH_K = 30
DEFAULT_THRESHOLD = 0.35
MMR_LAMBDA = 0.7  # 1.0 = pure relevance, 0.0 = pure diversity
MERGE_GAP_SECONDS = 20  # same-video hits closer than this get merged
RERANKER_MODEL = "BAAI/bge-reranker-base"  # slower but noticeably more accurate
SNIPPET_CHARS = 250

VALID_DOMAINS = (None, "dsa", "system_design")


class RetrievalError(Exception):
    """Clear, user-facing retrieval failure (Qdrant down, index missing, ...)."""


# --------------------------------------------------------------------------- #
# Result format
# --------------------------------------------------------------------------- #
@dataclass
class SearchResult:
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

    @classmethod
    def from_point(cls, point: Any, rank: int = 0) -> "SearchResult":
        """Build from a Qdrant point (score is cosine similarity, higher = better)."""
        p = point.payload or {}
        text = p.get("text", "")
        return cls(
            rank=rank,
            score=float(point.score),
            chunk_id=p.get("chunk_id", ""),
            video_id=p.get("video_id", ""),
            title=p.get("title", ""),
            channel=p.get("channel", ""),
            domain=p.get("domain", ""),
            start=int(p.get("start", 0)),
            end=int(p.get("end", 0)),
            start_label=p.get("start_label", ""),
            url=p.get("url", ""),
            text=text,
            snippet=text[:SNIPPET_CHARS],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Query cleaning
# --------------------------------------------------------------------------- #
def clean_query(query: str, glossary: dict[str, str] | None = None) -> str:
    """Strip, collapse whitespace, apply glossary fixes, truncate to 300 chars.

    Raises ValueError on an empty query so callers can report it clearly.
    """
    q = re.sub(r"\s+", " ", query or "").strip()
    if not q:
        raise ValueError("Query is empty - please type a question.")
    if glossary:
        for wrong, right in glossary.items():
            q = re.sub(re.escape(wrong), right, q, flags=re.IGNORECASE)
    return q[:MAX_QUERY_CHARS]


# --------------------------------------------------------------------------- #
# Post-processing (pure functions - easy to test)
# --------------------------------------------------------------------------- #
Pair = tuple[SearchResult, list[float]]  # result + its Qdrant vector


def apply_threshold(pairs: list[Pair], threshold: float) -> list[Pair]:
    """Drop results below the score threshold (empty list = 'no relevant content')."""
    return [(r, v) for r, v in pairs if r.score >= threshold]


def merge_overlaps(pairs: list[Pair], gap: int = MERGE_GAP_SECONDS) -> list[Pair]:
    """Merge hits from the same video that overlap or sit within `gap` seconds.

    Keeps the higher-scoring chunk as the base, merges the time range
    (min start, max end) and refreshes start_label/url to the merged start.
    """
    by_video: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_video.setdefault(pair[0].video_id, []).append(pair)

    merged: list[Pair] = []
    for video_pairs in by_video.values():
        video_pairs.sort(key=lambda p: p[0].start)
        current = video_pairs[0]
        for nxt in video_pairs[1:]:
            cur_res, _ = current
            nxt_res, _ = nxt
            if nxt_res.start <= cur_res.end + gap:  # overlapping / near-adjacent
                base = current if cur_res.score >= nxt_res.score else nxt
                br, bvec = base
                br.start = min(cur_res.start, nxt_res.start)
                br.end = max(cur_res.end, nxt_res.end)
                br.start_label = format_label(br.start)
                br.url = f"https://youtu.be/{br.video_id}?t={br.start}"
                current = (br, bvec)
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
    return merged


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of L2-normalized vectors == cosine similarity."""
    return sum(x * y for x, y in zip(a, b))


def mmr_select(pairs: list[Pair], top_k: int, lam: float = MMR_LAMBDA) -> list[Pair]:
    """Maximal marginal relevance: relevance (Qdrant's score) vs diversity.

    Prevents top_k from being 5 chunks of the same video. Doc-doc similarity
    is computed here (vectors are normalized); query-doc scores always come
    from Qdrant - never recomputed in Python.
    """
    remaining = list(pairs)
    selected: list[Pair] = []
    selected_vecs: list[list[float]] = []
    while remaining and len(selected) < top_k:
        best_i, best_val = 0, -math.inf
        for i, (res, vec) in enumerate(remaining):
            relevance = res.score  # cosine similarity from Qdrant
            diversity = max((_dot(vec, sv) for sv in selected_vecs), default=0.0)
            val = lam * relevance - (1.0 - lam) * diversity
            if val > best_val:
                best_val, best_i = val, i
        selected.append(remaining.pop(best_i))
        selected_vecs.append(selected[-1][1])
    return selected


def build_context(results: list[SearchResult]) -> str:
    """Numbered context blocks so the LLM can cite sources as [1], [2]..."""
    blocks = [
        f"[{i}] {r.title} @ {r.start_label} ({r.url})\n{r.text}"
        for i, r in enumerate(results, start=1)
    ]
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class Retriever:
    """Loads the embedding model and Qdrant client once; reuses them per query.

    model/client can be injected (tests pass fakes - no downloads, no network).
    """

    def __init__(
        self,
        model: SentenceTransformer | None = None,
        client: QdrantClient | None = None,
        glossary: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self._client = client
        self._reranker: Any = None
        self.glossary = load_glossary() if glossary is None else glossary
        self.last_timings: dict[str, float] = {"embed_ms": 0.0, "search_ms": 0.0}

    # -- lazy resources ---------------------------------------------------- #
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(MODEL_NAME, device=get_device())
        return self._model

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            # Only send the key over https (avoids a warning on local http Qdrant).
            api_key = QDRANT_API_KEY if QDRANT_URL.startswith("https") else None
            client = QdrantClient(url=QDRANT_URL, api_key=api_key, timeout=30)
            try:
                client.get_collections()
            except Exception as exc:
                raise RetrievalError(
                    f"Qdrant is not reachable at {QDRANT_URL} ({exc}).\n"
                    f"Start it with:\n  {QDRANT_DOCKER_CMD}"
                ) from exc
            self._client = client
        return self._client

    def _check_collection(self) -> None:
        """Fail early with a clear message if the index is missing/wrong."""
        client = self.client
        if not client.collection_exists(COLLECTION_NAME):
            raise RetrievalError(
                f"Collection '{COLLECTION_NAME}' does not exist. "
                f"Run first: python scripts/embed_and_store.py --domain all"
            )
        info = client.get_collection(COLLECTION_NAME)
        if info.points_count == 0:
            raise RetrievalError(
                f"Collection '{COLLECTION_NAME}' is empty. "
                f"Run first: python scripts/embed_and_store.py --domain all"
            )
        vectors = getattr(info.config.params, "vectors", None)
        size = getattr(vectors, "size", None)
        if size is not None and size != VECTOR_SIZE:
            raise RetrievalError(
                f"Collection vector size {size} != {VECTOR_SIZE} ({MODEL_NAME}). "
                f"Rebuild with: python scripts/embed_and_store.py --rebuild"
            )

    # -- embedding ---------------------------------------------------------- #
    def embed_query(self, query: str) -> list[float]:
        """Embed WITH the bge query prefix (documents were embedded without it)."""
        return self.model.encode(
            QUERY_PREFIX + query, normalize_embeddings=True
        ).tolist()

    # -- reranker (optional) ------------------------------------------------- #
    def _rerank(self, query: str, pairs: list[Pair], top_k: int) -> list[Pair]:
        """Cross-encoder rerank of the candidates.

        Slower than bi-encoder similarity alone, but noticeably more accurate.
        Lazy-loads the model only when --rerank is used.
        """
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(RERANKER_MODEL)
        texts = [r.text for r, _ in pairs]
        raw = self._reranker.predict([(query, t) for t in texts])
        rescored: list[Pair] = []
        for (res, vec), logit in zip(pairs, raw):
            res.score = 1.0 / (1.0 + math.exp(-float(logit)))  # sigmoid -> 0..1
            rescored.append((res, vec))
        rescored.sort(key=lambda p: p[0].score, reverse=True)
        return rescored[:top_k]

    # -- main search --------------------------------------------------------- #
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        fetch_k: int = DEFAULT_FETCH_K,
        domain: str | None = None,
        video_id: str | None = None,
        score_threshold: float = DEFAULT_THRESHOLD,
        use_mmr: bool = True,
        use_reranker: bool = False,
    ) -> list[SearchResult]:
        """Full pipeline: clean -> embed -> search -> threshold -> merge -> MMR/rerank.

        Returns [] when nothing passes the threshold (distinct from an error).
        """
        if domain not in VALID_DOMAINS:
            raise ValueError(f"domain must be one of {VALID_DOMAINS}, got {domain!r}")
        query = clean_query(query, self.glossary)
        self._check_collection()

        t0 = time.perf_counter()
        query_vec = self.embed_query(query)
        embed_ms = (time.perf_counter() - t0) * 1000

        conditions = []
        if domain:
            conditions.append(
                qm.FieldCondition(key="domain", match=qm.MatchValue(value=domain))
            )
        if video_id:
            conditions.append(
                qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id))
            )
        query_filter = qm.Filter(must=conditions) if conditions else None

        t0 = time.perf_counter()
        points = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=fetch_k,
            query_filter=query_filter,
            search_params=qm.SearchParams(hnsw_ef=SEARCH_HNSW_EF),
            score_threshold=score_threshold,  # server-side pre-filter
            with_payload=True,
            with_vectors=True,  # vectors needed for MMR diversity
        ).points
        search_ms = (time.perf_counter() - t0) * 1000
        self.last_timings = {"embed_ms": embed_ms, "search_ms": search_ms}

        pairs: list[Pair] = [
            (SearchResult.from_point(p), list(p.vector))
            for p in points
            if p.payload and p.vector is not None
        ]
        if not pairs:
            return []

        # Post-processing, in the specified order:
        pairs = apply_threshold(pairs, score_threshold)   # 1. threshold
        pairs = merge_overlaps(pairs)                     # 2. overlap merge
        if not pairs:
            return []
        if use_reranker:                                  # 3a. optional rerank
            pairs = self._rerank(query, pairs, top_k)
        elif use_mmr:                                     # 3b. diversity
            pairs = mmr_select(pairs, top_k)

        final = sorted((p[0] for p in pairs), key=lambda r: r.score, reverse=True)
        for i, res in enumerate(final, start=1):          # 4. rank + trim
            res.rank = i
        return final[:top_k]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the YouTube Reviser index.")
    parser.add_argument("query", help="the question to search for")
    parser.add_argument("--domain", choices=["dsa", "system_design"], default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K,
                        help="candidates fetched before post-processing")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="minimum cosine similarity (default 0.35)")
    parser.add_argument("--no-mmr", action="store_true",
                        help="disable diversity re-ranking (MMR)")
    parser.add_argument("--rerank", action="store_true",
                        help="use the bge cross-encoder (slower, more accurate)")
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    retriever = Retriever()
    try:
        results = retriever.search(
            args.query,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            domain=args.domain,
            score_threshold=args.threshold,
            use_mmr=not args.no_mmr,
            use_reranker=args.rerank,
        )
    except (ValueError, RetrievalError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)

    t = retriever.last_timings
    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        return

    print(f'query: "{args.query}"')
    print(f"embed: {t['embed_ms']:.0f}ms | search: {t['search_ms']:.0f}ms "
          f"| results: {len(results)}\n")
    if not results:
        print("no relevant content found (try a lower --threshold or drop --domain)")
        return
    for r in results:
        print(f"[{r.rank}] score={r.score:.3f}  {r.title}  @ {r.start_label}")
        print(f"     {r.url}")
        print(f"     {r.snippet}...\n")


if __name__ == "__main__":
    main()
