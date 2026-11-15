"use client";

import { tokenizeCitations } from "@/lib/citations";
import type { ChatMessage } from "@/lib/types";
import { CitationChip } from "./CitationChip";
import { ErrorNotice } from "./ErrorNotice";
import { TimestampList } from "./TimestampList";
import { TypingIndicator } from "./TypingIndicator";

interface MessageBubbleProps {
  message: ChatMessage;
  activeN: number | null;
  onSelectSource: (n: number) => void;
  onRetry: () => void;
}

export function MessageBubble({
  message,
  activeN,
  onSelectSource,
  onRetry,
}: MessageBubbleProps) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-sky-600 px-4 py-2.5 text-sm text-white shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  const valid = new Set(message.sources.map((source) => source.n));
  const parts = tokenizeCitations(message.content, valid);
  const sourceByNumber = new Map(
    message.sources.map((source) => [source.n, source]),
  );
  const isStreaming = message.status === "streaming";
  const hasContent = message.content.trim().length > 0;
  const showTyping = isStreaming && !hasContent;

  return (
    <div className="flex justify-start">
      <div
        aria-busy={isStreaming || undefined}
        className="w-full max-w-[95%] rounded-2xl rounded-bl-md border border-slate-200 bg-white px-4 py-3 text-sm shadow-sm dark:border-slate-700 dark:bg-slate-900"
      >
        <div
          className="text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500"
          aria-hidden="true"
        >
          Reviser
        </div>

        {showTyping ? <TypingIndicator /> : null}

        {message.content ? (
          <div
            aria-live="polite"
            className="mt-1 whitespace-pre-wrap break-words leading-relaxed text-slate-800 dark:text-slate-100"
          >
            {parts.map((part, index) => {
              if (part.type === "text") {
                return <span key={`text-${index}`}>{part.text}</span>;
              }
              const source = sourceByNumber.get(part.n);
              if (!source) {
                return <span key={`citation-${part.n}-${index}`}>[{part.n}]</span>;
              }
              return (
                <CitationChip
                  key={`citation-${part.n}-${index}`}
                  n={part.n}
                  source={source}
                  active={activeN === part.n}
                  onSelect={() => onSelectSource(part.n)}
                />
              );
            })}
          </div>
        ) : null}

        {message.status === "done" && message.found === false && !hasContent ? (
          <p className="mt-2 rounded-lg bg-slate-100 px-2.5 py-2 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300">
            This topic isn&apos;t in the indexed videos yet.
          </p>
        ) : null}

        {message.status === "done" &&
        message.found !== false &&
        message.uncited ? (
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
            This answer did not cite a source.
          </p>
        ) : null}

        {message.status === "cancelled" ? (
          <p className="mt-2 text-xs font-medium text-slate-500 dark:text-slate-400">
            Stopped.
          </p>
        ) : null}

        {message.error ? (
          <ErrorNotice
            error={message.error}
            onRetry={
              message.error.kind === "validation" ||
              message.error.kind === "auth"
                ? undefined
                : onRetry
            }
          />
        ) : null}

        {message.sources.length > 0 ? (
          <section
            aria-label="Sources for this answer"
            className="mt-3 border-t border-slate-100 pt-2 dark:border-slate-800"
          >
            <h3 className="text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
              Sources
            </h3>
            <TimestampList
              sources={message.sources}
              activeN={activeN}
              onSelect={onSelectSource}
            />
          </section>
        ) : null}

        {isStreaming ? (
          <span className="sr-only">Streaming answer…</span>
        ) : null}
      </div>
    </div>
  );
}
