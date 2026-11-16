# YouTube Reviser

A retrieval-augmented revision tool for placement preparation: ask about DSA or system design and get an answer grounded in indexed YouTube captions, with cited sources and timestamps that open at the relevant moment.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20search-DC244C)](https://qdrant.tech/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)](https://nextjs.org/)

> **Screenshot TODO:** Add a screenshot or GIF showing a cited answer, a clicked timestamp, and the embedded player.

**Stack:** Python | FastAPI | yt-dlp | youtube-transcript-api 1.x | sentence-transformers (`BAAI/bge-small-en-v1.5`) | Qdrant | Next.js | TypeScript | Tailwind CSS.

**Quick start:** Python 3.11+, Node.js 20.9+, and Docker are required. From the repository root, install dependencies, configure `.env`, start Qdrant, and build the local index:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up -d qdrant
python scripts/download_transcripts.py --domain all
python scripts/chunk_transcripts.py --domain all
python scripts/embed_and_store.py --domain all
```

Then start the API with `uvicorn app.main:app --reload --port 8000`; full Windows/Bash setup and frontend steps are in [Quick start](#5-quick-start).

## 2. Features

- Answers questions using indexed captions from [Padho with Pratyush](https://www.youtube.com/playlist?list=PLbJhGqY-mq47k_WLUtzVjmarUm1EuXPj2) DSA and Gaurav Sen system-design videos.
- Answers are prompted to stay under 250 words unless more detail is requested.
- Valid citation markers link to retrieved source videos and timestamps. Source links are assembled from retrieval results in code, not invented by the LLM.
- The API accepts a `domain` filter (`dsa` or `system_design`); the web chat searches both domains.
- `/ask/stream` streams answer tokens over Server-Sent Events (SSE). The web UI displays the stream.
- Clicking a source timestamp seeks the embedded YouTube player to that time. A separate link opens the original video.
- When retrieval returns no results, the answer is `No relevant content found in the indexed videos.` and the LLM is not called. If the retrieved context is insufficient, the model is instructed to say `I couldn't find this in the indexed videos.`

## 3. How it works

```text
Playlists
   |
   v
download_transcripts.py --> transcripts/<domain>/*.json
   |
   v
chunk_transcripts.py -----> chunks/*.jsonl
   |
   v
embed_and_store.py -------> sentence-transformers --> Qdrant
                                                       |
Question --> query embedding --> retrieve.py ----------+
                                                       |
                                                       v
                                         generate.py --> FastAPI --> Next.js UI
```

- **Dense retrieval:** document embeddings are normalized; queries use the BGE search prefix. Qdrant cosine search applies a score threshold and MMR diversity by default; an optional cross-encoder reranker is also implemented. There is no keyword/BM25 search.
- **Overlap handling:** adjacent hits from the same video are merged when their timestamps overlap or are within 20 seconds, reducing duplicate source entries.
- **Validated citations:** the LLM receives numbered context blocks. Invalid citation numbers are removed, and source links are built from the corresponding retrieved results.
- **No-context shortcut:** empty retrieval returns a fixed not-found response without spending an LLM call.
- **Direct SDKs:** generation uses provider SDKs in `scripts/generate.py`; it does not use LangChain.

Chunk windows target 78 seconds with 12 seconds of overlap. Chunk text is prefixed with the video title for embedding. Qdrant stores 384-dimensional cosine vectors, uses HNSW, and indexes `domain`, `video_id`, and `start` payload fields.

### Architecture and data pipeline

```mermaid
flowchart TD
    subgraph SRC["Sources"]
        A1["Padho with Pratyush<br/>DSA patterns playlist"]
        A2["Gaurav Sen<br/>system design playlist"]
    end

    subgraph ING["Ingestion - offline"]
        B1["youtube-transcript-api"]
        B2["yt-dlp subtitle fallback"]
        B3["Optional faster-whisper fallback<br/>DSA recovery pipeline"]
        C["transcripts/domain/video_id.json<br/>text, start, duration"]
        D["Clean and chunk<br/>78s windows, 12s overlap,<br/>title prefix"]
        E["chunks/domain.jsonl"]
        F["Embed with sentence-transformers<br/>bge-small-en-v1.5, 384 dims"]
    end

    G[("Qdrant<br/>HNSW, cosine,<br/>payload filter on domain")]

    subgraph API["FastAPI backend - root routes"]
        H["/search and /ask/stream"]
        I["Retriever<br/>query embedding with BGE prefix,<br/>score threshold, overlap merge, MMR"]
        J["LLM client<br/>Groq, OpenAI-compatible, or Anthropic"]
        K["Citation validation<br/>sources built from retrieval results"]
    end

    subgraph UI["Next.js frontend"]
        L["Chat panel<br/>streamed answer with citations"]
        M["YouTube player<br/>seek to timestamp"]
        N["Clickable timestamp list"]
    end

    A1 --> B1
    A2 --> B1
    B1 -->|"captions unavailable or blocked"| B2
    B2 --> C
    B3 -.->|"optional DSA recovery"| C
    B1 --> C
    C --> D
    D --> E
    E --> F
    F --> G

    L -->|"question"| H
    H --> I
    I -->|"query vector"| G
    G -->|"top chunks"| I
    I -->|"context blocks"| J
    I -.->|"no results: skip LLM"| H
    J --> K
    K -->|"SSE: tokens and sources"| L
    L --> N
    N -->|"click"| M
```

### What happens when a user asks a question

```mermaid
sequenceDiagram
    actor U as User
    participant UI as Next.js UI
    participant API as FastAPI
    participant R as Retriever
    participant Q as Qdrant
    participant L as LLM (Groq, OpenAI-compatible, or Anthropic)
    participant P as YouTube Player

    U->>UI: Ask "what is sliding window"
    UI->>API: POST /ask/stream
    API->>R: search(query, domain, top_k)
    R->>R: embed query with BGE prefix
    R->>Q: query points with domain filter
    Q-->>R: candidate chunks with cosine scores
    R->>R: threshold, overlap merge, MMR

    alt no relevant chunks
        R-->>API: empty list
        API-->>UI: "No relevant content found in the indexed videos."
    else chunks found
        R-->>API: numbered context blocks
        API->>L: system prompt, context, question
        L-->>API: streamed answer with citation markers
        API->>API: validate citations and build sources
        API-->>UI: SSE tokens, then sources event
        UI-->>U: answer with clickable citations
        U->>UI: click timestamp
        UI->>P: seek to timestamp
        P-->>U: video plays at that moment
    end
```

## 4. Current coverage

Counts below were read from the local Qdrant collection `youtube_reviser` on 2026-11-15.

| Domain | Indexed videos | Indexed chunks |
|---|---:|---:|
| DSA | 126 / 405* | 4,108 |
| System Design | 5 / 26 | 47 |

Some playlist videos are not indexed because caption requests have been blocked or rate-limited; the repository's failure logs include these errors. As a result, a topic may return the not-found response.

*The DSA denominator 405 is derived from 126 local transcript IDs plus 279 distinct queued IDs. This conflicts with the 316 total in the original README request; confirm the authoritative current playlist total before publishing. The System Design denominator 26 reconciles to 17 local transcript IDs plus 9 queued IDs.*

## 5. Quick start

### Prerequisites

- Python 3.11 or newer (`pyproject.toml` requires `>=3.11`; `.python-version` is `3.11`).
- Node.js 20.9 or newer (the installed Next.js package declares `>=20.9.0`).
- Docker Desktop with Docker Compose, and Git.
- A Groq API key for generated answers with the default `LLM_PROVIDER=groq`. `/search` works without an LLM key.

### 1. Clone and install Python dependencies

**PowerShell**

```powershell
git clone https://github.com/kxshith21/youtube-revisor-rag.git
Set-Location youtube-revisor-rag
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

**Bash**

```bash
git clone https://github.com/kxshith21/youtube-revisor-rag.git
cd youtube-revisor-rag
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY` before using `/ask`, `/ask/stream`, `/revise`, or `/quiz`. Do not commit `.env`.

### 2. Start Qdrant and build the index

From the repository root, start the Qdrant service:

```powershell
docker compose up -d qdrant
```

Run the pipeline (caption downloads can take a long time):

```powershell
python scripts/download_transcripts.py --domain all
python scripts/chunk_transcripts.py --domain all
python scripts/embed_and_store.py --domain all
```

The embedder downloads `BAAI/bge-small-en-v1.5` on first use. Qdrant data is stored in the Compose `qdrant_storage` volume.

### 3. Run the backend

In a new terminal at the repository root, activate `.venv` and run:

```powershell
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> for Swagger UI. API routes are currently at the root (for example, `/search`), not under `/api/v1`.

### 4. Run the frontend

In another terminal:

```powershell
if (-not (Test-Path web\.env.local)) { Copy-Item web\.env.example web\.env.local }
Set-Location web
npm install
npm run dev
```

Open <http://localhost:3000>. For Bash, use `[ -f web/.env.local ] || cp web/.env.example web/.env.local`, then `cd web`, `npm install`, and `npm run dev`. The web app requires the backend for real answers.

## 6. Configuration

Values below are the examples/defaults in `.env.example`. Never put real API keys in this README or in `web/.env.local`.

| Variable | Example value | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | Default LLM provider. The generator also has OpenAI-compatible/Ollama and Anthropic clients; optional provider packages may need separate installation. |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model name passed to the configured provider. |
| `GROQ_API_KEY` | *(blank)* | Required for the default Groq provider and generated answers. |
| `ANTHROPIC_API_KEY` | *(blank)* | Key for `LLM_PROVIDER=anthropic`. |
| `OPENAI_API_KEY` | *(blank)* | Key for `LLM_PROVIDER=openai` or an OpenAI-compatible provider; Ollama may run without one. |
| `OPENAI_BASE_URL` | *(blank)* | Optional custom endpoint, such as a local Ollama OpenAI-compatible endpoint. |
| `MAX_TOKENS` | `1000` | Maximum output tokens requested from the LLM. |
| `TEMPERATURE` | `0.2` | LLM sampling temperature. |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL. |
| `QDRANT_API_KEY` | *(blank)* | Optional Qdrant Cloud key; local Docker Qdrant needs none. |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:8501` | Comma-separated backend CORS origins. |
| `API_KEY` | *(blank)* | Optional API authentication key. `/health` remains public. |
| `RATE_LIMIT_SEARCH` | `30` | Search requests per minute, per IP. |
| `RATE_LIMIT_ASK` | `10` | Ask/revise/quiz requests per minute, per IP. |
| `LOG_LEVEL` | `INFO` | Backend log level. |
| `ASK_TIMEOUT_S` | `30` | Timeout for ask, revise, and quiz requests. |
| `STREAM_TIMEOUT_S` | `60` | Maximum SSE stream duration. |
| `STREAM_HEARTBEAT_S` | `15` | Interval between SSE heartbeat comments. |
| `CACHE_TTL_S` | `600` | In-memory cache lifetime for answers, topics, and videos. |
| `CACHE_MAX_ENTRIES` | `500` | Maximum entries per in-memory cache. |

`/search` performs retrieval only and does not require an LLM key. The current generator has Groq, OpenAI-compatible/Ollama, and Anthropic clients; Gemini is not implemented in `make_llm()`, despite `google-genai` appearing in `pyproject.toml`.

The frontend has a separate `web/.env.local`:

| Variable | Example value | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | FastAPI base URL. |
| `NEXT_PUBLIC_MOCK` | `false` | Set to `true` to show canned UI responses without the backend. |

## 7. Data pipeline

Run scripts from the repository root. `--domain` accepts `dsa`, `system_design`, or `all` where shown.

| Script | Input | Output | Example |
|---|---|---|---|
| `scripts/download_transcripts.py` | YouTube playlist captions | `transcripts/<domain>/<video_id>.json`; failure and missing queues | `python scripts/download_transcripts.py --domain dsa --limit 5` |
| `scripts/chunk_transcripts.py` | Transcript JSON and `glossary.json` | `chunks/dsa.jsonl` and `chunks/system_design.jsonl` | `python scripts/chunk_transcripts.py --domain all` |
| `scripts/embed_and_store.py` | Chunk JSONL | Points in Qdrant collection `youtube_reviser` | `python scripts/embed_and_store.py --domain all` |
| `scripts/retrieve.py` | Query and Qdrant collection | Ranked JSON or human-readable results | `python scripts/retrieve.py "sliding window" --domain dsa --top-k 5` |
| `scripts/generate.py` | Query, retrieved context, configured LLM | Answer/revision/quiz output with validated sources | `python scripts/generate.py "sliding window" --domain dsa --stream` |

`download_transcripts.py` uses `youtube-transcript-api` and falls back to yt-dlp captions. It supports `--retry-failed`, `--dry-run`, `--limit`, `--batch-size`, `--min-delay`, `--max-delay`, `--video-id`, `--priority-file`, and `--cookies-file`.

`scripts/download_playlist_full.py` is a separate **DSA-only** recovery pipeline. It tries the transcript API, yt-dlp captions, then optional faster-whisper transcription; it supports `--limit`, `--whisper-only`, `--captions-only`, `--whisper-model`, `--whisper-language`, `--priority-file`, and `--dry-run`.

`chunk_transcripts.py` supports `--window`, `--overlap`, and `--limit`. `embed_and_store.py` supports `--limit`, `--batch-size`, `--quantize`, and `--rebuild`. Use `--rebuild` when changing the embedding model; it deletes and recreates the collection, so rebuild both domains with:

```bash
python scripts/embed_and_store.py --domain all --rebuild
```

Other scripts include `scripts/query.py` (search), `scripts/dedupe_transcripts.py` (transcript/chunk deduplication), and `scripts/download_playlist_full.py` (DSA recovery).

### Repository layout

```text
.
app/                  # FastAPI app, services, schemas, routers, settings
scripts/              # Caption, chunk, embedding, retrieval, generation tools
tests/                # Python tests; external services are mocked
web/                  # Next.js, TypeScript, Tailwind chat UI
transcripts/          # Local downloaded captions; Git-ignored
chunks/               # Generated chunk JSONL; Git-ignored
logs/                 # Downloader failure reports
glossary.json          # Caption term corrections
Dockerfile             # API image
docker-compose.yml     # Qdrant and API services
requirements.txt       # Python runtime and test dependencies
pyproject.toml         # Python project metadata/dependencies
API.md                 # Expanded API, Docker, and curl documentation
```

## 8. API reference

The current routes are not prefixed with `/api/v1`. See Swagger at <http://localhost:8000/docs> or the expanded [API reference](API.md).

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service, Qdrant, embedding, and LLM readiness. |
| `POST` | `/search` | Retrieval only; no LLM call. |
| `POST` | `/ask` | Generate a grounded answer. |
| `POST` | `/ask/stream` | Stream an answer as SSE. |
| `POST` | `/revise` | Generate a revision note. |
| `POST` | `/quiz` | Generate cited quiz questions. |
| `GET` | `/topics` | Per-domain indexed video and chunk counts. |
| `GET` | `/videos` | Paginated indexed videos; supports `domain` and title query `q`. |

Examples use Bash `curl`. If `API_KEY` is set, add `-H "X-API-Key: $API_KEY"` to protected endpoints; `/health` is public.

**Search**

```bash
curl -sS -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"sliding window maximum","domain":"dsa","top_k":5,"use_mmr":true,"use_reranker":false}'
```

**Ask**

```bash
curl -sS -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"explain the sliding window pattern","domain":"dsa","top_k":5,"history":[]}'
```

**Ask with SSE streaming**

```bash
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"query":"explain the sliding window pattern","domain":"dsa","top_k":5,"history":[]}'
```

Pass `"domain": null` or omit `domain` to search both domains.

## 9. Evaluation

The checked-in Python tests exercise retrieval and citation behavior with fakes; no live question-to-video top-3 evaluation or timestamp-click results are recorded yet.

| Question | Correct video in top 3? | Timestamp link clicked and verified? |
|---|---|---|
| TODO: add a DSA evaluation question | TODO | TODO |
| TODO: add a System Design evaluation question | TODO | TODO |

## 10. Known limitations and troubleshooting

- **Incomplete coverage:** some caption requests receive YouTube `IpBlocked`, HTTP 429, or other retrieval errors. Wait before retrying; use a different network if appropriate; run smaller batches and increase `--min-delay`/`--max-delay`. `download_playlist_full.py` can fall back to faster-whisper for DSA. The transcript downloader does not expose a `--proxy` flag; proxy configuration is not verified by this project.
- **Qdrant unavailable:** start Docker Desktop, then run `docker compose up -d qdrant`. Check <http://localhost:6333/healthz>. If the API was started before Qdrant/index creation, restart the API.
- **Auto-caption terminology errors:** add wrong-to-right corrections to `glossary.json`, rerun chunking, then rebuild the Qdrant collection with `python scripts/embed_and_store.py --domain all --rebuild`. The embedder skips chunk IDs already in the collection, so a normal upsert run would not refresh existing chunks.
- **Embedding model mismatch:** do not mix vectors from different models. Rebuild the full collection with `python scripts/embed_and_store.py --domain all --rebuild`.
- **API returns 404 at `/`:** the backend has no homepage route. Use `/docs` for Swagger; the chat UI is on port 3000.
- **LLM key missing:** `/search` can still work. Add the configured provider's key to the repository-root `.env` and restart the backend to use `/ask`, streaming, revision, or quiz.

## 11. Roadmap

- Hybrid keyword plus vector retrieval.
- Flashcards and spaced repetition.
- Interview practice mode.
- Frame OCR for code or diagrams shown in videos.
- Add an `/api/v1` prefix; current endpoints are root-level.
- Add a Gemini SDK adapter; the dependency is declared, but the generator does not currently implement Gemini.

The cross-encoder reranker is already implemented as an optional retrieval mode; it is not listed as a future feature.

## 12. Legal and attribution

Video content belongs to its original creators: [Padho with Pratyush](https://www.youtube.com/playlist?list=PLbJhGqY-mq47k_WLUtzVjmarUm1EuXPj2) and [Gaurav Sen](https://www.youtube.com/playlist?list=PLMCXHnjXnTnvo6alSjVkgxV-VH6EPyvoX). YouTube Reviser is an independent learning tool and is not affiliated with either creator. Answers link back to the original videos.

The downloader creates local transcript files and the chunker creates local chunk files; these are not part of the application source. `transcripts/` and `chunks/` are Git-ignored, and users generate them locally by running the downloader and pipeline.

**Please watch the original videos** and support their creators.

## 13. Contributing, license, and acknowledgements

Contributions are welcome. Open an issue describing the problem or proposed change, then submit a focused pull request. Run `python -m pytest` for backend tests; in `web/`, run `npm test`, `npm run lint`, and `npm run typecheck`.

**License:** MIT selected; a `LICENSE` file is not present yet. Add the MIT license text with the copyright holder before distributing the project.

**Acknowledgements:** Padho with Pratyush and Gaurav Sen for the educational videos; the maintainers of FastAPI, Qdrant, sentence-transformers, and Next.js.

## TODO: verify before publishing

- Add the demo screenshot or GIF.
- Confirm the DSA playlist denominator: local data indicates 405 IDs, while the initial request specified 316.
- Update the DSA downloader scripts to use the Padho with Pratyush playlist URL; their source URL has not yet been changed.
- Add the MIT `LICENSE` file with the owner's name and year.
- Add the maintainer's name and GitHub/LinkedIn or project links.
- Run a real evaluation set and record whether expected videos appear in the top three and timestamp links work.
- Gemini is not wired into the current generator; implement or remove the unused `google-genai` dependency if Gemini support is intended.
