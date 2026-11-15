"use client";

import { cn } from "@/lib/cn";
import type { DomainFilter } from "@/lib/types";

const OPTIONS: Array<{ value: DomainFilter; label: string; hint: string }> = [
  { value: "all", label: "All", hint: "Search both domains" },
  { value: "dsa", label: "DSA", hint: "Padho with Pratyush DSA videos" },
  {
    value: "system_design",
    label: "System Design",
    hint: "Gaurav Sen videos",
  },
];

interface DomainToggleProps {
  value: DomainFilter;
  onChange: (value: DomainFilter) => void;
  disabled?: boolean;
}

export function DomainToggle({
  value,
  onChange,
  disabled = false,
}: DomainToggleProps) {
  return (
    <div
      role="radiogroup"
      aria-label="Domain filter"
      className="inline-flex rounded-lg border border-slate-200 bg-slate-100/70 p-0.5 dark:border-slate-700 dark:bg-slate-800/70"
    >
      {OPTIONS.map((option) => {
        const checked = value === option.value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={checked}
            aria-label={option.hint}
            disabled={disabled}
            onClick={() => onChange(option.value)}
            className={cn(
              "rounded-md px-2.5 py-1.5 text-xs font-medium transition",
              "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-500",
              "disabled:cursor-not-allowed disabled:opacity-50",
              checked
                ? "bg-white text-slate-900 shadow-sm dark:bg-slate-950 dark:text-white"
                : "text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-100",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
