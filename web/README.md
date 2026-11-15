# YouTube Reviser Web

Next.js chat UI for the FastAPI backend in the repository root. It streams RAG
answers over SSE, renders citation chips, and plays the cited YouTube moment in
an embedded player.

## Setup

```bash
cd web
npm install
cp .env.example .env.local   # Windows: copy .env.example .env.local
npm run dev
```

Open <http://localhost:3000>.

The backend must be running for real answers:

```bash
# from the repository root
uvicorn app.main:app --reload --port 8000
```

## Environment

`web/.env.local`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_MOCK=false
```

- `NEXT_PUBLIC_MOCK=true` streams canned tokens and sources, so the UI works
  without Qdrant, an LLM key, or the backend. Ask about "bread" to see the
  `found=false` state.
- Never put `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` in this
  file. Only `NEXT_PUBLIC_*` values are exposed to the browser.

## Scripts

```bash
npm run dev        # dev server on :3000
npm run build      # production build + type check
npm run start      # serve the production build
npm run lint       # ESLint
npm run typecheck  # tsc --noEmit
npm test           # Vitest (SSE parser, citations, timestamps)
npm run test:watch
```

## CORS

The backend allows `http://localhost:3000` by default. To be explicit, set this
in the repository-root `.env` and restart uvicorn:

```bash
ALLOWED_ORIGINS=http://localhost:3000
```

If Next starts on another port (3001), add that origin too. Use `localhost`
consistently — `127.0.0.1` is a different origin.

The backend also exposes `Retry-After`, which lets the UI show
"Too many requests. Try again in Ns." on HTTP 429.

## Structure

| File | Purpose |
|---|---|
| `app/page.tsx` | Page shell, hydration/persistence, player state |
| `components/ChatPanel.tsx` | Message list, auto-scroll, composer |
| `components/MessageBubble.tsx` | Answer text, citations, sources, errors |
| `components/ChatInput.tsx` | Multiline input, Enter/Shift+Enter, Stop |
| `components/DomainToggle.tsx` | All / DSA / System Design |
| `components/VideoPlayer.tsx` | YouTube embed, embed-error notice |
| `components/TimestampList.tsx` | Keyboard-navigable source rows |
| `components/CitationChip.tsx` | Inline `[n]` button |
| `hooks/useChatStream.ts` | Fetch SSE, abort, retries, error mapping |
| `hooks/useYouTubePlayer.ts` | IFrame API lifecycle and seeking |
| `lib/api.ts` | HTTP client, SSE decode, `ApiError` |
| `lib/sse.ts` | Incremental SSE parser |
| `lib/citations.ts` | `[1]` / `[2][3]` tokenizer |
| `lib/time.ts` | `mm:ss` / `h:mm:ss` |
| `lib/youtube.ts` | Source URL parsing, embed loader |
| `lib/store.ts` | Zustand store (messages, domain, active source) |
| `lib/storage.ts` | localStorage persistence (guarded) |
| `lib/mock.ts` | Mock streaming API |

## Manual test checklist

1. **Ask** — “explain the sliding window pattern” with DSA selected. Typing dots
   appear, tokens stream, citations render, sources are listed.
2. **Same-video timestamps** — ask a broad question, then click two sources from
   the same video. The player seeks without reloading.
3. **Different video** — switch to All or ask a system-design question, then
   click a source from another video. The iframe loads the new video at `t`.
4. **Cancel** — ask a long question and press **Stop** mid-stream. Partial text
   stays, the bubble says “Stopped.”, and the input is enabled again.
5. **Off-topic** — ask “how to bake bread”. The answer shows the not-found
   notice, no sources, and no player.
6. **Mobile (375px)** — one column, no horizontal scroll, player card appears
   above the chat after a timestamp tap, close button works.
7. **Errors** — `RATE_LIMIT_ASK=1` and ask twice for the 429 countdown; stop the
   API for the 503 retry; use an unavailable video for the embed notice and
   “Open on YouTube”.
8. **Persistence/theme/a11y** — reload restores the chat; **New chat** clears it;
   the theme follows the system by default; Tab reaches every control and
   arrow keys move through timestamp rows.
