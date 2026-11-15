"use client";

import { useEffect, useState } from "react";
import type { ChatError } from "@/lib/types";

interface ErrorNoticeProps {
  error: ChatError;
  onRetry?: () => void;
}

function describe(error: ChatError, remaining: number): string {
  switch (error.kind) {
    case "rate_limit":
      return remaining > 0
        ? `Too many requests. Try again in ${remaining}s.`
        : "Too many requests. You can retry now.";
    case "unavailable":
      return `The backend isn't ready yet. ${error.message}`;
    case "network":
      return "Couldn't reach the API. Make sure the backend is running at http://localhost:8000.";
    case "timeout":
      return error.message || "The request timed out.";
    case "auth":
    case "validation":
    case "server":
    case "stream":
    default:
      return error.message;
  }
}

export function ErrorNotice({ error, onRetry }: ErrorNoticeProps) {
  const [remaining, setRemaining] = useState(() =>
    Math.max(0, Math.ceil(error.retryAfterS ?? 0)),
  );

  useEffect(() => {
    if (error.kind !== "rate_limit") return;

    const timer = window.setInterval(() => {
      setRemaining((value) => (value <= 0 ? 0 : value - 1));
    }, 1000);

    return () => window.clearInterval(timer);
  }, [error.kind]);

  return (
    <div
      role="alert"
      className="mt-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950 dark:border-amber-700/60 dark:bg-amber-950/40 dark:text-amber-100"
    >
      <p>{describe(error, remaining)}</p>

      {error.requestId ? (
        <p className="mt-1 text-xs text-amber-800/80 dark:text-amber-200/70">
          Request ID: <code className="font-mono">{error.requestId}</code>
        </p>
      ) : null}

      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          disabled={remaining > 0}
          className="mt-2 inline-flex items-center rounded-md border border-amber-400 px-2.5 py-1 text-xs font-semibold transition hover:bg-amber-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-600 disabled:cursor-not-allowed disabled:opacity-50 dark:border-amber-600 dark:hover:bg-amber-900/60"
        >
          Retry
        </button>
      ) : null}
    </div>
  );
}
