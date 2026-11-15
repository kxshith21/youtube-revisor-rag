"""Shared constants for all YouTube Reviser scripts.

One source of truth so embed_and_store.py, query.py, retrieve.py, etc.
can never drift apart.
"""

from __future__ import annotations

import os
import uuid

# Load .env (if present) so os.getenv() below picks up local secrets.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv optional; defaults still work
    pass
# --------------------------------------------------------------------------- #
# Embedding model
# --------------------------------------------------------------------------- #
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384  # bge-small-en-v1.5 output dimensions

# Queries get this prefix; documents are embedded WITHOUT it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None  # optional (Qdrant Cloud)
COLLECTION_NAME = "youtube_reviser"

# Collection creation settings (named so they are easy to tune).
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
SEARCH_HNSW_EF = 128  # query-time ef

# Deterministic point IDs: uuid5(NAMESPACE_URL, chunk_id) -> stable across runs.
NAMESPACE_URL = uuid.NAMESPACE_URL

# --------------------------------------------------------------------------- #
# LLM (Phase 6: generate.py, and the backend in Phase 7)
# --------------------------------------------------------------------------- #
# Provider/model come from .env so they can be swapped without code changes.
# Check the provider's docs for current model names (Groq: console.groq.com/docs/models).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Pick the key belonging to the configured provider. Falling back to any key
# would silently send e.g. a Groq key to OpenAI when several keys are present.
_PROVIDER_NAME = (LLM_PROVIDER or "groq").lower()
if _PROVIDER_NAME == "groq":
    LLM_API_KEY = os.getenv("GROQ_API_KEY") or None
elif _PROVIDER_NAME == "anthropic":
    LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY") or None
elif _PROVIDER_NAME in ("openai", "openai-compatible"):
    LLM_API_KEY = os.getenv("OPENAI_API_KEY") or None
else:  # ollama or an unknown/custom provider
    LLM_API_KEY = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or None
    )

# --------------------------------------------------------------------------- #
# Hint shown when Qdrant is unreachable. The container usually already exists,
# so `docker start` first - re-running `docker run` would fail on a name conflict.
# --------------------------------------------------------------------------- #
QDRANT_DOCKER_CMD = (
    "docker start qdrant    # container already exists\n"
    "  # (first time only) docker run -d --name qdrant -p 6333:6333 "
    "-v qdrant_storage:/qdrant/storage qdrant/qdrant"
)
