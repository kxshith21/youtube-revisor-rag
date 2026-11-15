"""Quick similarity search against the Qdrant collection (step 3 sanity test).

Usage (from the project root):
    python scripts/query.py "what is sliding window" --domain dsa --top-k 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

# Shared constants (MODEL_NAME, QUERY_PREFIX, QDRANT_URL, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    COLLECTION_NAME,
    MODEL_NAME,
    QDRANT_API_KEY,
    QDRANT_DOCKER_CMD,
    QDRANT_URL,
    QUERY_PREFIX,
    SEARCH_HNSW_EF,
)
from embed_and_store import get_device  # noqa: E402


def embed_query(model: SentenceTransformer, query: str) -> list[float]:
    """Embed a query WITH the bge prefix (documents were embedded without it)."""
    return model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()


def search(
    client: QdrantClient,
    vector: list[float],
    top_k: int,
    domain: str | None,
) -> list[Any]:
    """query_points with cosine similarity scores + optional domain filter."""
    query_filter = None
    if domain:
        query_filter = qm.Filter(
            must=[qm.FieldCondition(key="domain", match=qm.MatchValue(value=domain))]
        )
    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        query_filter=query_filter,
        search_params=qm.SearchParams(hnsw_ef=SEARCH_HNSW_EF),
        with_payload=True,
        with_vectors=False,
    ).points


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the YouTube Reviser index.")
    parser.add_argument("query", help="the question to search for")
    parser.add_argument("--domain", choices=["dsa", "system_design"], default=None,
                        help="restrict results to one domain")
    parser.add_argument("--top-k", type=int, default=5, help="number of results (default: 5)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    device = get_device()
    print(f"model:  {MODEL_NAME} ({device})  |  qdrant: {QDRANT_URL}\n")

    # Only send the key over https (avoids a warning on local http Qdrant).
    api_key = QDRANT_API_KEY if QDRANT_URL.startswith("https") else None
    client = QdrantClient(url=QDRANT_URL, api_key=api_key, timeout=30)
    try:
        client.get_collections()
    except Exception as exc:
        print(f"ERROR: cannot reach Qdrant at {QDRANT_URL} ({exc}).")
        print("Start it with:")
        print(f"  {QDRANT_DOCKER_CMD}")
        sys.exit(1)
    if not client.collection_exists(COLLECTION_NAME):
        print(f"ERROR: collection '{COLLECTION_NAME}' does not exist.")
        print("Run first: python scripts/embed_and_store.py --domain all")
        sys.exit(1)

    model = SentenceTransformer(MODEL_NAME, device=device)

    t0 = time.perf_counter()
    vector = embed_query(model, args.query)
    t_embed = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    results = search(client, vector, args.top_k, args.domain)
    t_search = (time.perf_counter() - t0) * 1000

    print(f'query: "{args.query}"  ({t_embed:.0f}ms embed, {t_search:.0f}ms search)\n')
    if not results:
        print("no results (try a lower --top-k or drop --domain)")
        return

    for rank, point in enumerate(results, start=1):
        p = point.payload
        # Qdrant returns cosine similarity directly for Distance.COSINE (higher = better).
        print(f"[{rank}] score={point.score:.3f}  {p['title']}  @ {p['start_label']}")
        print(f"     {p['url']}")
        print(f"     {p['text'][:200]}...\n")


if __name__ == "__main__":
    main()
