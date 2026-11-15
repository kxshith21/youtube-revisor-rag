/**
 * Types mirroring the FastAPI schemas in `app/schemas.py`.
 *
 * Keep this file in sync with the backend. Everything is structural (no
 * classes) so responses can be validated at runtime before they reach state.
 */

export type Domain = "dsa" | "system_design";
/** `all` is a UI-only value that maps to `domain: null` in requests. */
export type DomainFilter = Domain | "all";

export interface HistoryItem {
  question: string;
  answer: string;
}

export interface AskRequest {
  query: string;
  domain: Domain | null;
  top_k: number;
  history: HistoryItem[];
}

export interface SourceOut {
  n: number;
  title: string;
  start_label: string;
  url: string;
  domain: string;
  score: number;
}

export interface AnswerOut {
  query: string;
  answer: string;
  sources: SourceOut[];
  found: boolean;
  uncited: boolean;
  model: string;
  latency_ms: Record<string, number>;
  usage: Record<string, number>;
}

export interface ErrorEnvelope {
  error: {
    code: string;
    message: string;
    request_id: string;
  };
}

/** Events yielded by the client-side SSE reader (already JSON-decoded). */
export type StreamEvent =
  | { type: "token"; text: string }
  | { type: "done"; answer: AnswerOut }
  | { type: "error"; message: string };

export type MessageStatus = "streaming" | "done" | "error" | "cancelled";

export type ChatErrorKind =
  | "network"
  | "rate_limit"
  | "unavailable"
  | "timeout"
  | "server"
  | "stream"
  | "validation"
  | "auth";

export interface ChatError {
  kind: ChatErrorKind;
  message: string;
  /** Seconds from the `Retry-After` header, if the server sent one. */
  retryAfterS?: number | null;
  requestId?: string | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: SourceOut[];
  found: boolean | null;
  uncited: boolean;
  status: MessageStatus;
  error: ChatError | null;
  createdAt: number;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
