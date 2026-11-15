"use client";

import { useEffect, useRef } from "react";

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  isStreaming: boolean;
}

const MAX_LENGTH = 300;

export function ChatInput({
  value,
  onChange,
  onSubmit,
  onStop,
  isStreaming,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 160)}px`;
  }, [value]);

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.shiftKey) return;
    if (event.nativeEvent.isComposing) return; // IME composition
    event.preventDefault();
    onSubmit();
  }

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
      className="border-t border-slate-200 bg-white/90 p-3 backdrop-blur dark:border-slate-700 dark:bg-slate-900/90"
    >
      <label htmlFor="chat-input" className="sr-only">
        Ask a question
      </label>

      <div className="flex items-end gap-2 rounded-xl border border-slate-300 bg-white px-3 py-2 focus-within:border-sky-500 focus-within:ring-2 focus-within:ring-sky-500/20 dark:border-slate-600 dark:bg-slate-950">
        <textarea
          ref={textareaRef}
          id="chat-input"
          value={value}
          rows={1}
          maxLength={MAX_LENGTH}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about DSA or system design…"
          aria-describedby="chat-input-hint"
          className="max-h-40 min-h-6 flex-1 resize-none bg-transparent py-1 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:text-slate-100 dark:placeholder:text-slate-500"
        />

        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-slate-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-500 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-300"
          >
            <span aria-hidden="true">■</span> Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!value.trim()}
            className="inline-flex shrink-0 items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-sky-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-500 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
          >
            Send <span aria-hidden="true">↑</span>
          </button>
        )}
      </div>

      <div
        id="chat-input-hint"
        className="mt-1.5 flex items-center justify-between gap-2 px-1 text-[11px] text-slate-400 dark:text-slate-500"
      >
        <span>Enter to send · Shift+Enter for a new line</span>
        <span className="tabular-nums">
          {value.length}/{MAX_LENGTH}
        </span>
      </div>
    </form>
  );
}
