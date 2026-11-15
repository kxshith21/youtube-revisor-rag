"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ChatPanel } from "@/components/ChatPanel";
import { ThemeToggle } from "@/components/ThemeToggle";
import { TimestampList } from "@/components/TimestampList";
import { VideoPlayer } from "@/components/VideoPlayer";
import { useChatStream } from "@/hooks/useChatStream";
import { cn } from "@/lib/cn";
import { clearChat, loadChat, saveChat } from "@/lib/storage";
import { useChatStore } from "@/lib/store";
import type { ChatMessage } from "@/lib/types";
import { parseSourceVideo } from "@/lib/youtube";

export default function Home() {
  const messages = useChatStore((state) => state.messages);
  const activeSource = useChatStore((state) => state.activeSource);
  const hydrated = useChatStore((state) => state.hydrated);
  const selectSource = useChatStore((state) => state.selectSource);
  const clearSource = useChatStore((state) => state.clearSource);
  const reset = useChatStore((state) => state.reset);
  const hydrate = useChatStore((state) => state.hydrate);
  const { send, stop, isStreaming } = useChatStream();

  const [seekKey, setSeekKey] = useState(0);

  // Restore once, after mount, so server and client render the same shell.
  useEffect(() => {
    const saved = loadChat();
    hydrate(saved?.messages ?? [], "all");
  }, [hydrate]);

  // Debounced persistence; every write is guarded inside storage.ts.
  useEffect(() => {
    if (!hydrated) return;
    const timer = window.setTimeout(() => saveChat(messages, "all"), 250);
    return () => window.clearTimeout(timer);
  }, [messages, hydrated]);

  const activeMessage = useMemo(() => {
    if (!activeSource) return null;
    return (
      messages.find((message) => message.id === activeSource.messageId) ?? null
    );
  }, [activeSource, messages]);

  const active = useMemo(() => {
    if (!activeMessage || !activeSource) return null;
    return (
      activeMessage.sources.find((source) => source.n === activeSource.n) ??
      null
    );
  }, [activeMessage, activeSource]);

  const video = useMemo(
    () => (active ? parseSourceVideo(active.url) : null),
    [active],
  );

  const handleSelect = useCallback(
    (messageId: string, n: number) => {
      selectSource(messageId, n);
      setSeekKey((key) => key + 1);
    },
    [selectSource],
  );

  const handleNewChat = useCallback(() => {
    stop();
    reset();
    clearChat();
    setSeekKey(0);
  }, [reset, stop]);

  const handleRetry = useCallback(
    (message: ChatMessage) => {
      const state = useChatStore.getState();
      const index = state.messages.findIndex(
        (entry) => entry.id === message.id,
      );
      const userMessage = index > 0 ? state.messages[index - 1] : null;
      if (!userMessage || userMessage.role !== "user") return;

      state.removeMessage(message.id);
      void send(userMessage.content, { addUser: false });
    },
    [send],
  );

  return (
    <div className="mx-auto flex h-dvh w-full max-w-7xl flex-col gap-3 p-3 sm:gap-4 sm:p-4">
      <header className="flex flex-wrap items-center justify-between gap-2.5">
        <div className="flex min-w-0 items-center gap-2.5">
          <div
            aria-hidden="true"
            className="grid size-9 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-sky-500 to-indigo-600 text-sm font-bold text-white shadow-sm"
          >
            ▶
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-base font-semibold text-slate-900 dark:text-white">
              YouTube Reviser
            </h1>
            <p className="hidden truncate text-xs text-slate-500 dark:text-slate-400 sm:block">
              Grounded answers from your indexed DSA and system design videos
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleNewChat}
            className="inline-flex h-9 items-center rounded-lg border border-slate-200 px-2.5 text-xs font-medium text-slate-600 transition hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-500 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
          >
            New chat
          </button>

          <ThemeToggle />
        </div>
      </header>

      <main className="flex min-h-0 flex-1 flex-col gap-3 lg:flex-row">
        <aside
          aria-label="Video and sources"
          className={cn(
            "order-1 flex shrink-0 flex-col gap-3 lg:order-2 lg:w-[400px]",
            active && video
              ? "max-lg:max-h-[48dvh] max-lg:overflow-y-auto"
              : "hidden lg:flex",
          )}
        >
          {active && video ? (
            <>
              <VideoPlayer
                videoId={video.videoId}
                start={video.start}
                title={active.title}
                seekKey={seekKey}
                onClose={clearSource}
              />

              {activeMessage && activeMessage.sources.length > 0 ? (
                <div className="hidden min-h-0 flex-col overflow-y-auto rounded-2xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-700 dark:bg-slate-900 lg:flex">
                  <h2 className="text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
                    Sources in this answer
                  </h2>
                  <TimestampList
                    sources={activeMessage.sources}
                    activeN={activeSource?.n ?? null}
                    onSelect={(n) => handleSelect(activeMessage.id, n)}
                  />
                </div>
              ) : null}
            </>
          ) : (
            <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-slate-300 bg-white/60 p-6 text-center dark:border-slate-700 dark:bg-slate-900/40">
              <div
                aria-hidden="true"
                className="grid size-11 place-items-center rounded-xl bg-slate-100 text-lg text-slate-500 dark:bg-slate-800 dark:text-slate-400"
              >
                ▶
              </div>
              <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
                Video moments appear here
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Ask a question, then click a citation or timestamp to jump to
                that exact moment.
              </p>
            </div>
          )}
        </aside>

        <section
          aria-label="Chat"
          className="order-2 flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 shadow-sm dark:border-slate-700 dark:bg-slate-950/60 lg:order-1"
        >
          <ChatPanel
            messages={messages}
            activeSource={activeSource}
            onSend={(query) => void send(query)}
            onStop={stop}
            onRetry={handleRetry}
            onSelectSource={handleSelect}
            isStreaming={isStreaming}
          />
        </section>
      </main>
    </div>
  );
}
