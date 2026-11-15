"use client";

import { useCallback, useRef, useState } from "react";
import { ApiError, streamAnswer } from "@/lib/api";
import { makeId } from "@/lib/ids";
import { useChatStore } from "@/lib/store";
import type {
  ChatError,
  ChatMessage,
  DomainFilter,
  HistoryItem,
} from "@/lib/types";

const MAX_HISTORY = 3; // backend accepts at most 3 Q/A pairs

function buildHistory(messages: ChatMessage[]): HistoryItem[] {
  const pairs: HistoryItem[] = [];
  for (let index = 0; index < messages.length - 1; index += 1) {
    const question = messages[index];
    const answer = messages[index + 1];
    if (
      question.role === "user" &&
      answer.role === "assistant" &&
      answer.status === "done"
    ) {
      pairs.push({
        question: question.content.slice(0, 500),
        answer: answer.content.slice(0, 4000),
      });
    }
  }
  return pairs.slice(-MAX_HISTORY);
}

function toChatError(error: unknown): ChatError {
  if (error instanceof ApiError) {
    if (error.status === 429) {
      return {
        kind: "rate_limit",
        message: error.message,
        retryAfterS: error.retryAfterS ?? 60,
        requestId: error.requestId,
      };
    }
    if (error.status === 503) {
      return {
        kind: "unavailable",
        message: error.message,
        requestId: error.requestId,
      };
    }
    if (error.status === 401 || error.status === 403) {
      return {
        kind: "auth",
        message: "The API rejected this request. Check the X-API-Key setting.",
        requestId: error.requestId,
      };
    }
    if (error.status === 422) {
      return {
        kind: "validation",
        message: error.message,
        requestId: error.requestId,
      };
    }
    if (error.status === 504) {
      return {
        kind: "timeout",
        message: "The request timed out. Try a shorter question.",
        requestId: error.requestId,
      };
    }
    return {
      kind: "server",
      message: error.message,
      requestId: error.requestId,
    };
  }

  return {
    kind: "network",
    message:
      "Couldn't reach the API. Make sure it is running at http://localhost:8000.",
  };
}

export interface SendOptions {
  /** Retry reuses the existing user bubble instead of adding a duplicate. */
  addUser?: boolean;
  domain?: DomainFilter;
}

export function useChatStream() {
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const streamingIdRef = useRef<string | null>(null);

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const send = useCallback(
    async (rawQuery: string, options: SendOptions = {}) => {
      const query = rawQuery.trim();
      if (!query || streamingIdRef.current) return;

      const store = useChatStore.getState();
      const domain = options.domain ?? store.domain;
      const history = buildHistory(store.messages);

      if (options.addUser !== false) {
        store.addMessage({
          id: makeId(),
          role: "user",
          content: query,
          sources: [],
          found: null,
          uncited: false,
          status: "done",
          error: null,
          createdAt: Date.now(),
        });
      }

      const assistantId = makeId();
      store.addMessage({
        id: assistantId,
        role: "assistant",
        content: "",
        sources: [],
        found: null,
        uncited: false,
        status: "streaming",
        error: null,
        createdAt: Date.now(),
      });

      streamingIdRef.current = assistantId;
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        for await (const event of streamAnswer(
          {
            query,
            domain: domain === "all" ? null : domain,
            top_k: 5,
            history,
          },
          controller.signal,
        )) {
          if (controller.signal.aborted) break;
          const state = useChatStore.getState();

          if (event.type === "token") {
            state.appendToken(assistantId, event.text);
          } else if (event.type === "done") {
            state.updateMessage(assistantId, {
              content: event.answer.answer,
              sources: event.answer.sources,
              found: event.answer.found,
              uncited: event.answer.uncited,
              status: "done",
              error: null,
            });
          } else if (event.type === "error") {
            state.updateMessage(assistantId, {
              status: "error",
              error: { kind: "stream", message: event.message },
            });
          }
        }

        const finalMessage = useChatStore
          .getState()
          .messages.find((message) => message.id === assistantId);

        if (finalMessage?.status === "streaming") {
          useChatStore.getState().updateMessage(
            assistantId,
            controller.signal.aborted
              ? { status: "cancelled" }
              : {
                  status: "error",
                  error: {
                    kind: "server",
                    message: "The stream ended before an answer was received.",
                  },
                },
          );
        }
      } catch (error) {
        const state = useChatStore.getState();
        if (controller.signal.aborted) {
          state.updateMessage(assistantId, { status: "cancelled" });
        } else {
          state.updateMessage(assistantId, {
            status: "error",
            error: toChatError(error),
          });
        }
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
        streamingIdRef.current = null;
        setIsStreaming(false);
      }
    },
    [],
  );

  return { send, stop, isStreaming };
}
