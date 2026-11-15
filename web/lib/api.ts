/**
 * HTTP client for the FastAPI backend.
 *
 * `/ask/stream` is a POST endpoint, so EventSource cannot be used; this module
 * reads the response body with `fetch` + ReadableStream and feeds the SSE
 * parser. Non-2xx responses are converted into a typed ApiError that keeps the
 * request id and Retry-After seconds for the UI.
 */

import { mockStream } from "./mock";
import { flushSSE, parseSSEChunk, type SSERawEvent } from "./sse";
import {
  isRecord,
  type AnswerOut,
  type AskRequest,
  type SourceOut,
  type StreamEvent,
} from "./types";

const DEFAULT_BASE_URL = "http://localhost:8000";

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL || DEFAULT_BASE_URL
).replace(/\/+$/, "");

export function isMockMode(): boolean {
  return process.env.NEXT_PUBLIC_MOCK === "true";
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;
  readonly retryAfterS: number | null;

  constructor(
    status: number,
    code: string,
    message: string,
    requestId: string | null = null,
    retryAfterS: number | null = null,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.retryAfterS = retryAfterS;
  }
}

function parseRetryAfter(value: string | null): number | null {
  if (!value) return null;

  const seconds = Number(value.trim());
  if (Number.isFinite(seconds) && seconds >= 0) return Math.ceil(seconds);

  const date = Date.parse(value);
  if (!Number.isNaN(date)) {
    return Math.max(0, Math.ceil((date - Date.now()) / 1000));
  }
  return null;
}

async function toApiError(response: Response): Promise<ApiError> {
  let code = "http_error";
  let message = `The API request failed (HTTP ${response.status}).`;
  let requestId = response.headers.get("X-Request-ID");

  try {
    const body: unknown = await response.json();
    if (isRecord(body) && isRecord(body.error)) {
      if (typeof body.error.code === "string") code = body.error.code;
      if (typeof body.error.message === "string") message = body.error.message;
      if (typeof body.error.request_id === "string") {
        requestId = body.error.request_id;
      }
    }
  } catch {
    // Non-JSON error body (proxy, gateway): keep the generic message.
  }

  return new ApiError(
    response.status,
    code,
    message,
    requestId,
    parseRetryAfter(response.headers.get("Retry-After")),
  );
}

function normalizeSource(value: unknown): SourceOut | null {
  if (!isRecord(value)) return null;
  const { n, title, start_label, url, domain, score } = value;
  if (
    typeof n !== "number" ||
    typeof title !== "string" ||
    typeof start_label !== "string" ||
    typeof url !== "string" ||
    typeof domain !== "string" ||
    typeof score !== "number"
  ) {
    return null;
  }
  return { n, title, start_label, url, domain, score };
}

function normalizeNumberMap(value: unknown): Record<string, number> {
  if (!isRecord(value)) return {};
  const result: Record<string, number> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (typeof entry === "number" && Number.isFinite(entry)) result[key] = entry;
  }
  return result;
}

export function normalizeAnswer(value: unknown): AnswerOut | null {
  if (!isRecord(value)) return null;
  const { query, answer, sources, found, uncited, model } = value;
  if (
    typeof query !== "string" ||
    typeof answer !== "string" ||
    !Array.isArray(sources) ||
    typeof found !== "boolean" ||
    typeof uncited !== "boolean" ||
    typeof model !== "string"
  ) {
    return null;
  }

  const normalizedSources = sources
    .map(normalizeSource)
    .filter((source): source is SourceOut => source !== null);

  return {
    query,
    answer,
    sources: normalizedSources,
    found,
    uncited,
    model,
    latency_ms: normalizeNumberMap(value.latency_ms),
    usage: normalizeNumberMap(value.usage),
  };
}

export function decodeEvent(raw: SSERawEvent): StreamEvent | null {
  let payload: unknown;
  try {
    payload = JSON.parse(raw.data);
  } catch {
    return raw.event === "error"
      ? { type: "error", message: "The server sent a malformed event." }
      : null;
  }

  if (raw.event === "token") {
    if (isRecord(payload) && typeof payload.text === "string") {
      return payload.text ? { type: "token", text: payload.text } : null;
    }
    return null;
  }

  if (raw.event === "error") {
    const message =
      isRecord(payload) && typeof payload.message === "string"
        ? payload.message
        : "The answer stream failed.";
    return { type: "error", message };
  }

  if (raw.event === "done") {
    const answer = normalizeAnswer(payload);
    return answer
      ? { type: "done", answer }
      : { type: "error", message: "The server returned an invalid answer." };
  }

  return null; // `end`, heartbeats, and unknown events are ignored
}

/**
 * Stream a RAG answer. Yields decoded events until `done`, `error`, or EOF.
 * Throws ApiError for HTTP failures and DOMException on abort.
 */
export async function* streamAnswer(
  body: AskRequest,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  if (isMockMode()) {
    yield* mockStream(body, signal);
    return;
  }

  const response = await fetch(`${API_BASE_URL}/ask/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
    cache: "no-store",
  });

  if (!response.ok) throw await toApiError(response);
  if (!response.body) {
    throw new ApiError(
      response.status,
      "empty_stream",
      "The server returned an empty stream.",
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const parsed = parseSSEChunk(
        buffer,
        decoder.decode(value, { stream: true }),
      );
      buffer = parsed.rest;
      for (const raw of parsed.events) {
        const event = decodeEvent(raw);
        if (event) yield event;
      }
    }

    buffer += decoder.decode();
    for (const raw of flushSSE(buffer)) {
      const event = decodeEvent(raw);
      if (event) yield event;
    }
  } finally {
    reader.releaseLock();
  }
}
