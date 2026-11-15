# YouTube Reviser API

FastAPI wrapper around the existing retrieval (`scripts/retrieve.py`) and
generation (`scripts/generate.py`) modules. It loads the embedding model,
Qdrant client, and LLM client once at startup.

Run everything from the project root so `scripts/config.py` can read `.env`.

## Prerequisites

1. Copy `.env.example` to `.env` and fill in the provider key.
2. Start Qdrant and populate the collection:
   ```bash
   docker start qdrant        # or: docker run -d --name qdrant -p 6333:6333 \
                              #       -v qdrant_storage:/qdrant/storage qdrant/qdrant
   python scripts/embed_and_store.py --domain all
   ```

## Run

Development:

```bash
uvicorn app.main:app --reload --port 8000
```

Production (one worker is recommended because the model, cache, and rate
limiter are in-process):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 \
  --proxy-headers --no-access-log
```

If `API_KEY` is set in `.env`, add `-H "X-API-Key: $API_KEY"` to every request
except `/health`.

## Swagger docs

With the server running, open:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
- OpenAPI JSON: <http://localhost:8000/openapi.json>

Click **Authorize** in Swagger when `API_KEY` is configured and paste the key.
Request/response schemas include examples, and `/ask/stream` documents the SSE
event format.

## curl examples

### Health

```bash
curl -s http://localhost:8000/health
```

### Search (retrieval only, no LLM)

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "sliding window maximum",
    "domain": "dsa",
    "top_k": 5,
    "use_mmr": true,
    "use_reranker": false
  }'
```

### Ask (full RAG answer)

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "explain the sliding window pattern",
    "domain": "dsa",
    "top_k": 5,
    "history": []
  }'
```

The response includes `answer`, validated `sources`, `found`, `uncited`,
`model`, `latency_ms`, and `usage`. Repeated cacheable questions return
`X-Cache: HIT`.

### Ask (streaming SSE)

```bash
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "query": "explain the sliding window pattern",
    "domain": "dsa",
    "top_k": 5
  }'
```

Events:

```
event: token
data: {"text": "Sliding window is ..."}

event: done
data: {"query": "...", "answer": "...", "sources": [...], ...}
```

`event: error` may replace `done` if generation fails. Comment heartbeats
(`: heartbeat`) are sent every 15 seconds.

### Revise

```bash
curl -s -X POST http://localhost:8000/revise \
  -H "Content-Type: application/json" \
  -d '{"topic": "binary search", "domain": "dsa"}'
```

### Quiz

```bash
curl -s -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{"topic": "consistent hashing", "domain": "system_design", "num_questions": 5}'
```

Malformed LLM JSON is retried once by `generate.py`; a second failure returns
HTTP 502.

### Topics

```bash
curl -s http://localhost:8000/topics
```

### Videos

```bash
curl -s "http://localhost:8000/videos?domain=dsa&q=sliding&limit=20&offset=0"
```

### With API key authentication

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"query": "binary search", "domain": "dsa"}'
```

## Docker

The compose file starts the API and Qdrant, with persistent volumes for Qdrant
storage and the Hugging Face/embedding model cache.

```bash
cp .env.example .env          # fill in keys
docker compose up -d qdrant   # start the Qdrant volume first
python scripts/embed_and_store.py --domain all
docker compose up -d --build api
```

Notes:

- Stop any existing local Qdrant container first if it already uses port 6333.
- The compose Qdrant volume is separate from a previously created Docker
  volume, so the index must be populated against the compose Qdrant instance
  (the command above writes to `localhost:6333`, exposed by compose).
- If the API was started before the index was populated, restart it:
  `docker compose restart api`.
- The API is at <http://localhost:8000>; Swagger at
  <http://localhost:8000/docs>.

## Errors

Every non-2xx response uses:

```json
{
  "error": {
    "code": "index_unavailable",
    "message": "Qdrant is not reachable at http://localhost:6333.",
    "request_id": "..."
  }
}
```

Common codes: `validation_error` (422), `unauthorized` (401),
`rate_limited` (429), `llm_rate_limited` (429), `llm_failed` (502),
`quiz_parse_error` (502), `index_unavailable` (503), `llm_unavailable` (503),
`timeout` (504), `internal_error` (500).

Every response also includes `X-Request-ID`.

## Frontend

The chat UI lives in [`web/`](web/README.md) (Next.js on port 3000). It calls
`POST /ask/stream` with `fetch` because SSE here is a POST endpoint.

Allow the dev origin in the repository-root `.env` (it is the default):

```bash
ALLOWED_ORIGINS=http://localhost:3000
```

CORS also exposes `Retry-After` (alongside `X-Request-ID` and `X-Cache`), so the
browser can show the 429 countdown instead of a generic error. Restart uvicorn
after changing `.env`. See [web/README.md](web/README.md) for setup, mock mode,
and the manual test checklist.
