"use client";

import { useRef } from "react";
import { cn } from "@/lib/cn";
import type { SourceOut } from "@/lib/types";
import { parseSourceVideo, watchUrl } from "@/lib/youtube";

interface TimestampListProps {
  sources: SourceOut[];
  activeN: number | null;
  onSelect: (n: number) => void;
}

function domainLabel(domain: string): string {
  if (domain === "system_design") return "System Design";
  if (domain === "dsa") return "DSA";
  return domain;
}

export function TimestampList({
  sources,
  activeN,
  onSelect,
}: TimestampListProps) {
  const buttonRefs = useRef<Array<HTMLButtonElement | null>>([]);

  function moveFocus(
    event: React.KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) {
    const keys = ["ArrowDown", "ArrowUp", "Home", "End"];
    if (!keys.includes(event.key)) return;

    event.preventDefault();
    let next = index;
    if (event.key === "ArrowDown") next = (index + 1) % sources.length;
    if (event.key === "ArrowUp") {
      next = (index - 1 + sources.length) % sources.length;
    }
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = sources.length - 1;

    buttonRefs.current[next]?.focus();
  }

  return (
    <ul aria-label="Sources with timestamps" className="mt-2 space-y-1.5">
      {sources.map((source, index) => {
        const video = parseSourceVideo(source.url);
        const active = source.n === activeN;

        return (
          <li key={`${source.n}-${source.url}`} className="flex gap-1">
            <button
              ref={(element) => {
                buttonRefs.current[index] = element;
              }}
              type="button"
              disabled={!video}
              onClick={() => onSelect(source.n)}
              onKeyDown={(event) => moveFocus(event, index)}
              aria-current={active ? "true" : undefined}
              aria-label={`Play source ${source.n}: ${source.title} at ${source.start_label}`}
              className={cn(
                "group flex min-w-0 flex-1 items-center gap-2 rounded-lg border px-2.5 py-2 text-left text-xs transition",
                "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-500",
                "disabled:cursor-not-allowed disabled:opacity-50",
                active
                  ? "border-sky-500 bg-sky-50 dark:border-sky-500 dark:bg-sky-950/50"
                  : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:hover:bg-slate-800",
              )}
            >
              <span
                className={cn(
                  "grid size-5 shrink-0 place-items-center rounded text-[10px] font-semibold",
                  active
                    ? "bg-sky-600 text-white"
                    : "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300",
                )}
              >
                {source.n}
              </span>

              <span className="min-w-0 flex-1">
                <span className="block truncate font-medium text-slate-800 dark:text-slate-100">
                  {source.title}
                </span>
                <span className="mt-0.5 flex items-center gap-1.5 text-[11px] text-slate-500 dark:text-slate-400">
                  <span className="tabular-nums">{source.start_label}</span>
                  <span aria-hidden="true">·</span>
                  <span>{domainLabel(source.domain)}</span>
                </span>
              </span>

              <span
                aria-hidden="true"
                className="shrink-0 text-slate-400 transition group-hover:text-sky-600 dark:group-hover:text-sky-400"
              >
                ▶
              </span>
            </button>

            <a
              href={video ? watchUrl(video.videoId, video.start) : source.url}
              target="_blank"
              rel="noreferrer"
              aria-label={`Open source ${source.n} on YouTube`}
              title="Open on YouTube"
              className="grid w-8 shrink-0 place-items-center rounded-lg border border-slate-200 text-slate-500 transition hover:bg-slate-50 hover:text-slate-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-500 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-white"
            >
              <span aria-hidden="true">↗</span>
            </a>
          </li>
        );
      })}
    </ul>
  );
}
