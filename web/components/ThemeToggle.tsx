"use client";

import { useSyncExternalStore } from "react";
import { useTheme } from "next-themes";
import { cn } from "@/lib/cn";

const ORDER = ["system", "light", "dark"] as const;
type ThemeName = (typeof ORDER)[number];

function isThemeName(value: string | undefined): value is ThemeName {
  return value === "system" || value === "light" || value === "dark";
}

const ICONS: Record<ThemeName, string> = {
  system: "💻",
  light: "☀️",
  dark: "🌙",
};

const subscribe = () => () => {};

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  // false on the server/first paint, true after hydration — no effect needed.
  const mounted = useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );

  const current: ThemeName = isThemeName(theme) ? theme : "system";
  // Before mount the server rendered "system", while `theme` is already the
  // client's stored value. Every theme-derived output must use this fallback
  // until mounted, or hydration mismatches.
  const activeTheme: ThemeName = mounted ? current : "system";
  const next = ORDER[(ORDER.indexOf(activeTheme) + 1) % ORDER.length];

  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      aria-label={`Theme: ${activeTheme}. Switch to ${next} theme.`}
      title={`Theme: ${activeTheme} (click for ${next})`}
      className={cn(
        "inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 px-2.5 text-xs font-medium",
        "text-slate-600 transition hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-500",
        "dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800",
      )}
    >
      <span aria-hidden="true">{ICONS[activeTheme]}</span>
      <span className="hidden capitalize sm:inline">{activeTheme}</span>
    </button>
  );
}
