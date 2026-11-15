"use client";

import { cn } from "@/lib/cn";
import type { SourceOut } from "@/lib/types";

interface CitationChipProps {
  n: number;
  source: SourceOut;
  active: boolean;
  onSelect: () => void;
}

export function CitationChip({
  n,
  source,
  active,
  onSelect,
}: CitationChipProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-current={active ? "true" : undefined}
      aria-label={`Source ${n}: ${source.title} at ${source.start_label}`}
      title={`${source.title} — ${source.start_label}`}
      className={cn(
        "mx-0.5 inline-flex items-center rounded px-1 align-super text-[0.7em] font-semibold leading-none",
        "transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-500",
        active
          ? "bg-sky-600 text-white"
          : "bg-sky-100 text-sky-800 hover:bg-sky-200 dark:bg-sky-950 dark:text-sky-300 dark:hover:bg-sky-900",
      )}
    >
      {n}
    </button>
  );
}
