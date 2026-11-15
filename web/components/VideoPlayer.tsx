"use client";

import { useYouTubePlayer } from "@/hooks/useYouTubePlayer";
import { formatTimestamp } from "@/lib/time";
import { watchUrl } from "@/lib/youtube";

interface VideoPlayerProps {
  videoId: string;
  start: number;
  title: string;
  seekKey: number;
  onClose?: () => void;
}

export function VideoPlayer({
  videoId,
  start,
  title,
  seekKey,
  onClose,
}: VideoPlayerProps) {
  const { containerRef, ready, error } = useYouTubePlayer({
    videoId,
    start,
    seekKey,
  });

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <div className="flex items-start justify-between gap-2 border-b border-slate-200 p-3 dark:border-slate-700">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-slate-800 dark:text-slate-100">
            {title || "YouTube video"}
          </p>
          <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
            Playing from {formatTimestamp(start)}
          </p>
        </div>

        {onClose ? (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close video player"
            className="grid size-8 shrink-0 place-items-center rounded-lg text-slate-500 transition hover:bg-slate-100 hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-500 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-white"
          >
            <span aria-hidden="true">✕</span>
          </button>
        ) : null}
      </div>

      <div className="relative aspect-video bg-black">
        <div
          aria-hidden={error ? true : undefined}
          className={error ? "hidden" : "h-full w-full"}
        >
          <div ref={containerRef} className="h-full w-full" />
        </div>

        {!ready && !error ? (
          <p className="absolute inset-0 grid place-items-center text-xs text-slate-300">
            Loading player…
          </p>
        ) : null}

        {error ? (
          <div
            role="alert"
            className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-4 text-center"
          >
            <p className="text-sm text-slate-100">{error}</p>
            <a
              href={watchUrl(videoId, start)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 rounded-lg bg-white px-3 py-1.5 text-xs font-semibold text-slate-900 transition hover:bg-slate-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-400"
            >
              Open on YouTube <span aria-hidden="true">↗</span>
            </a>
          </div>
        ) : null}
      </div>
    </div>
  );
}
