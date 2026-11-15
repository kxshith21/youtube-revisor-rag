"use client";

import { useEffect, useRef, useState } from "react";
import type { ActiveSource } from "@/lib/store";
import type { ChatMessage } from "@/lib/types";
import { ChatInput } from "./ChatInput";
import { MessageBubble } from "./MessageBubble";

interface ChatPanelProps {
  messages: ChatMessage[];
  activeSource: ActiveSource | null;
  onSend: (query: string) => void;
  onStop: () => void;
  onRetry: (message: ChatMessage) => void;
  onSelectSource: (messageId: string, n: number) => void;
  isStreaming: boolean;
}

const SAMPLE_PROMPTS = [
  "Explain the sliding window pattern",
  "How does consistent hashing work?",
  "Binary search vs linear search",
];

export function ChatPanel({
  messages,
  activeSource,
  onSend,
  onStop,
  onRetry,
  onSelectSource,
  isStreaming,
}: ChatPanelProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const pinnedRef = useRef(true);
  const [showJump, setShowJump] = useState(false);
  const [input, setInput] = useState("");

  useEffect(() => {
    const element = scrollRef.current;
    if (!element || !pinnedRef.current) return;
    element.scrollTop = element.scrollHeight;
  }, [messages]);

  function handleScroll() {
    const element = scrollRef.current;
    if (!element) return;
    const distance =
      element.scrollHeight - element.scrollTop - element.clientHeight;
    const pinned = distance < 80;
    pinnedRef.current = pinned;
    setShowJump((current) => (current === !pinned ? current : !pinned));
  }

  function jumpToLatest() {
    const element = scrollRef.current;
    if (!element) return;
    element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
    pinnedRef.current = true;
    setShowJump(false);
  }

  function submit() {
    const query = input.trim();
    if (!query || isStreaming) return;
    setInput("");
    pinnedRef.current = true;
    setShowJump(false);
    onSend(query);
  }

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        role="log"
        aria-relevant="additions text"
        aria-label="Conversation"
        className="flex-1 space-y-4 overflow-y-auto px-3 py-4 sm:px-4"
      >
        {messages.length === 0 ? (
          <div className="mx-auto flex max-w-md flex-col items-center gap-4 py-10 text-center">
            <div
              aria-hidden="true"
              className="grid size-12 place-items-center rounded-2xl bg-sky-100 text-xl text-sky-700 dark:bg-sky-950 dark:text-sky-300"
            >
              ▶
            </div>
            <div>
              <h2 className="text-base font-semibold text-slate-900 dark:text-white">
                Ask your video library
              </h2>
              <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                Answers cite the exact moment in the indexed DSA and system
                design videos.
              </p>
            </div>

            <ul className="flex w-full flex-col gap-2">
              {SAMPLE_PROMPTS.map((prompt) => (
                <li key={prompt}>
                  <button
                    type="button"
                    onClick={() => setInput(prompt)}
                    className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-left text-xs text-slate-700 transition hover:border-sky-400 hover:bg-sky-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
                  >
                    {prompt}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              activeN={
                activeSource?.messageId === message.id
                  ? activeSource.n
                  : null
              }
              onSelectSource={(n) => onSelectSource(message.id, n)}
              onRetry={() => onRetry(message)}
            />
          ))
        )}
      </div>

      {showJump ? (
        <button
          type="button"
          onClick={jumpToLatest}
          className="absolute bottom-24 left-1/2 -translate-x-1/2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-md transition hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
        >
          Jump to latest ↓
        </button>
      ) : null}

      <ChatInput
        value={input}
        onChange={setInput}
        onSubmit={submit}
        onStop={onStop}
        isStreaming={isStreaming}
      />
    </div>
  );
}
