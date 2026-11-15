/**
 * Conversation persistence.
 *
 * Every call is wrapped in try/catch: localStorage can be unavailable (SSR,
 * privacy mode) or full (quota), and a chat page must keep working either way.
 */

import { makeId } from "./ids";
import {
  isRecord,
  type ChatError,
  type ChatErrorKind,
  type ChatMessage,
  type DomainFilter,
  type MessageStatus,
  type SourceOut,
} from "./types";

const STORAGE_KEY = "yt-revisor:chat:v1";
const STORAGE_VERSION = 1;
const MAX_MESSAGES = 60;

interface PersistedChat {
  version: number;
  domain: DomainFilter;
  messages: ChatMessage[];
}

const ERROR_KINDS: ChatErrorKind[] = [
  "network",
  "rate_limit",
  "unavailable",
  "timeout",
  "server",
  "stream",
  "validation",
  "auth",
];

function asDomain(value: unknown): DomainFilter {
  return value === "dsa" || value === "system_design" ? value : "all";
}

function asSource(value: unknown): SourceOut | null {
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

function asError(value: unknown): ChatError | null {
  if (!isRecord(value)) return null;
  if (
    typeof value.kind !== "string" ||
    !ERROR_KINDS.includes(value.kind as ChatErrorKind) ||
    typeof value.message !== "string"
  ) {
    return null;
  }
  return {
    kind: value.kind as ChatErrorKind,
    message: value.message,
    retryAfterS:
      typeof value.retryAfterS === "number" ? value.retryAfterS : null,
    requestId:
      typeof value.requestId === "string" ? value.requestId : null,
  };
}

function asStatus(value: unknown): MessageStatus {
  if (value === "done" || value === "error" || value === "cancelled") {
    return value;
  }
  // A reload cannot resume a stream, so an interrupted answer becomes stopped.
  return "cancelled";
}

function asMessage(value: unknown): ChatMessage | null {
  if (!isRecord(value)) return null;
  if (value.role !== "user" && value.role !== "assistant") return null;
  if (typeof value.content !== "string") return null;

  const sources = Array.isArray(value.sources)
    ? value.sources
        .map(asSource)
        .filter((source): source is SourceOut => source !== null)
    : [];

  return {
    id: typeof value.id === "string" && value.id ? value.id : makeId(),
    role: value.role,
    content: value.content,
    sources,
    found: typeof value.found === "boolean" ? value.found : null,
    uncited: value.uncited === true,
    status: asStatus(value.status),
    error: asError(value.error),
    createdAt:
      typeof value.createdAt === "number" ? value.createdAt : Date.now(),
  };
}

export function loadChat(): {
  messages: ChatMessage[];
  domain: DomainFilter;
} | null {
  if (typeof window === "undefined") return null;

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;

    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed) || parsed.version !== STORAGE_VERSION) return null;
    if (!Array.isArray(parsed.messages)) return null;

    const messages = parsed.messages
      .map(asMessage)
      .filter((message): message is ChatMessage => message !== null)
      .slice(-MAX_MESSAGES);

    return { messages, domain: asDomain(parsed.domain) };
  } catch {
    return null;
  }
}

export function saveChat(
  messages: ChatMessage[],
  domain: DomainFilter,
): void {
  if (typeof window === "undefined") return;

  try {
    const payload: PersistedChat = {
      version: STORAGE_VERSION,
      domain,
      messages: messages.slice(-MAX_MESSAGES),
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Quota exceeded or storage disabled: keep the session in memory.
  }
}

export function clearChat(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Nothing else to do.
  }
}
