"use client";

export function TypingIndicator() {
  return (
    <div
      role="status"
      aria-label="Generating answer"
      className="flex items-center gap-1.5 py-1"
    >
      {[0, 1, 2].map((index) => (
        <span
          key={index}
          aria-hidden="true"
          className="size-2 animate-bounce rounded-full bg-slate-400 motion-reduce:animate-none dark:bg-slate-500"
          style={{ animationDelay: `${index * 140}ms` }}
        />
      ))}
      <span className="sr-only">Generating answer…</span>
    </div>
  );
}
