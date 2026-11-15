"""Step 3 of the YouTube Reviser RAG pipeline.

Reads chunks/*.jsonl, embeds them with sentence-transformers, and stores the
vectors in a persistent Qdrant collection.

Usage (from the project root):
    python scripts/embed_and_store.py --domain all --limit 50   # test run
    python scripts/embed_and_store.py --domain all              # full run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Shared constants (MODEL_NAME, QUERY_PREFIX, QDRANT_URL, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    COLLECTION_NAME,
    HNSW_EF_CONSTRUCT,
    HNSW_M,
    MODEL_NAME,
    NAMESPACE_URL,
    QDRANT_API_KEY,
    QDRANT_DOCKER_CMD,
    QDRANT_URL,
    VECTOR_SIZE,
)

ROOT = Path(__file__).resolve().parents[1]  # project root, CWD-independent
CHUNKS_DIR = ROOT / "chunks"
DOMAINS = ("dsa", "system_design")
DEFAULT_BATCH_SIZE = 64


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_chunks(domain: str, limit: int | None) -> list[dict[str, Any]]:
    """Read chunks/<domain>.jsonl; skip malformed lines and empty text."""
    path = CHUNKS_DIR / f"{domain}.jsonl"
    if not path.exists():
        print(f"  WARNING: {path.relative_to(ROOT)} not found, skipping domain")
        return []

    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if not chunk.get("text") or not chunk.get("text_for_embedding"):
                    raise ValueError("empty text/text_for_embedding")
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"  WARNING: skipping malformed line {lineno} of {path.name} ({exc})")
                continue
            chunks.append(chunk)
            if limit is not None and len(chunks) >= limit:
                break
    return chunks


# --------------------------------------------------------------------------- #
# Model & device
# --------------------------------------------------------------------------- #
def get_device() -> str:
    """Pick GPU (cuda/mps) if available, else CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_model(device: str) -> SentenceTransformer:
    """Load the embedding model once (downloads on first use)."""
    return SentenceTransformer(MODEL_NAME, device=device)


def embed_batch(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Embed one batch of documents (no prefix, normalized vectors)."""
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #
def get_client() -> QdrantClient:
    """Connect to Qdrant; exit with a clear message if it is not running."""
    # Only send the key over https (avoids a warning on local http Qdrant).
    api_key = QDRANT_API_KEY if QDRANT_URL.startswith("https") else None
    client = QdrantClient(url=QDRANT_URL, api_key=api_key, timeout=30)
    try:
        client.get_collections()
    except Exception as exc:
        print(f"ERROR: cannot reach Qdrant at {QDRANT_URL} ({exc}).")
        print("Is it running? Start it with:")
        print(f"  {QDRANT_DOCKER_CMD}")
        sys.exit(1)
    return client


def make_point_id(chunk_id: str) -> str:
    """Deterministic uuid5 from the chunk_id (Qdrant needs int/UUID, not str)."""
    return str(uuid.uuid5(NAMESPACE_URL, chunk_id))


def ensure_collection(
    client: QdrantClient, rebuild: bool, quantize: bool
) -> None:
    """Create the collection if missing; guard against model/size mismatches."""
    exists = client.collection_exists(COLLECTION_NAME)
    if exists and rebuild:
        print(f"  --rebuild: deleting collection '{COLLECTION_NAME}'")
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if exists:
        info = client.get_collection(COLLECTION_NAME)
        # Mixing embeddings from two models breaks search -> demand --rebuild.
        params = info.config.params or {}
        vector_size = None
        vectors = getattr(params, "vectors", None)
        if vectors is not None:
            vector_size = getattr(vectors, "size", None)
        if vector_size is not None and vector_size != VECTOR_SIZE:
            print(
                f"ERROR: collection vector size {vector_size} != {VECTOR_SIZE} "
                f"({MODEL_NAME}). Re-run with --rebuild to rebuild from scratch."
            )
            sys.exit(1)
        # Model name lives in the collection metadata; check it.
        stored_model = (info.config.metadata or {}).get("model")
        if stored_model and stored_model != MODEL_NAME:
            print(
                f"ERROR: collection was built with a different model "
                f"({stored_model!r} != {MODEL_NAME!r}). Re-run with --rebuild "
                f"(mixing embeddings from two models breaks search)."
            )
            sys.exit(1)
        if quantize:
            _enable_quantization(client)
        return

    # New collection: cosine distance + HNSW tuning + model marker.
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qm.VectorParams(size=VECTOR_SIZE, distance=qm.Distance.COSINE),
        hnsw_config=qm.HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
        metadata={"model": MODEL_NAME},
    )
    # Payload indexes so filtered search (domain/video_id/start) stays fast.
    for field, schema in (
        ("domain", qm.PayloadSchemaType.KEYWORD),
        ("video_id", qm.PayloadSchemaType.KEYWORD),
        ("start", qm.PayloadSchemaType.INTEGER),
    ):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )
    if quantize:
        _enable_quantization(client)
    print(f"  created collection '{COLLECTION_NAME}' (cosine, {VECTOR_SIZE}d)")


def _enable_quantization(client: QdrantClient) -> None:
    """Optional int8 scalar quantization for lower memory use."""
    client.update_collection(
        collection_name=COLLECTION_NAME,
        quantization_config=qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationParams(
                type=qm.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        ),
    )
    print("  enabled int8 scalar quantization")


def existing_ids(client: QdrantClient, ids: list[str]) -> set[str]:
    """Return the subset of point ids already stored (for resumability)."""
    found: set[str] = set()
    for i in range(0, len(ids), 256):  # retrieve in batches
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[make_point_id(x) for x in ids[i : i + 256]],
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        found.update(p.payload["chunk_id"] for p in points if p.payload)
    return found


def count_domain(client: QdrantClient, domain: str) -> int:
    """Point count for one domain (filtered count)."""
    return client.count(
        collection_name=COLLECTION_NAME,
        count_filter=qm.Filter(
            must=[qm.FieldCondition(key="domain", match=qm.MatchValue(value=domain))]
        ),
    ).count


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed chunks/*.jsonl and store them in Qdrant."
    )
    parser.add_argument("--domain", "-domain", choices=["dsa", "system_design", "all"],
                        default="all", help="which domain to embed (default: all)")
    parser.add_argument("--rebuild", action="store_true",
                        help="delete and recreate the collection from scratch")
    parser.add_argument("--quantize", action="store_true",
                        help="enable int8 scalar quantization (lower memory)")
    parser.add_argument("--limit", "-limit", type=int, default=None, metavar="N",
                        help="only embed the first N chunks per domain (for testing)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"embed/upsert batch size (default: {DEFAULT_BATCH_SIZE})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.time()
    domains = list(DOMAINS) if args.domain == "all" else [args.domain]

    device = get_device()
    print(f"device: {device}")
    print(f"model:  {MODEL_NAME}")
    print(f"qdrant: {QDRANT_URL}")

    client = get_client()
    ensure_collection(client, rebuild=args.rebuild, quantize=args.quantize)

    model = load_model(device)  # load once, reuse for every batch

    read = skipped = embedded = 0
    for domain in domains:
        chunks = load_chunks(domain, args.limit)
        read += len(chunks)

        # Resumable: skip chunks already stored (unless --rebuild wiped them).
        stored = existing_ids(client, list({c["chunk_id"] for c in chunks})) if chunks else set()
        todo = [c for c in chunks if c["chunk_id"] not in stored]
        skipped += len(chunks) - len(todo)
        print(f"  [{domain}] read={len(chunks)} already stored={len(chunks) - len(todo)} "
              f"to embed={len(todo)}")

        # Embed a batch, then upsert it: a crash only loses one batch.
        for i in tqdm(range(0, len(todo), args.batch_size), desc=f"embed {domain}"):
            batch = todo[i : i + args.batch_size]
            vectors = embed_batch(model, [c["text_for_embedding"] for c in batch])
            points = [
                qm.PointStruct(
                    id=make_point_id(c["chunk_id"]),
                    vector=vec,
                    payload={
                        "chunk_id": c["chunk_id"],
                        "video_id": c["video_id"],
                        "title": c["title"],
                        "channel": c.get("channel", ""),
                        "domain": c["domain"],
                        "chunk_index": int(c["chunk_index"]),
                        "start": int(c["start"]),
                        "end": int(c["end"]),
                        "start_label": c["start_label"],
                        "url": c["url"],
                        "text": c["text"],
                    },
                )
                for c, vec in zip(batch, vectors)
            ]
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            embedded += len(points)

    # Reporting
    total = client.count(collection_name=COLLECTION_NAME).count
    print("\n=== summary ===")
    print(f"chunks read:   {read}")
    print(f"chunks skipped: {skipped} (already stored)")
    print(f"chunks embedded: {embedded}")
    print(f"total points in collection: {total}")
    for domain in domains:
        print(f"  {domain}: {count_domain(client, domain)}")
    print(f"elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
