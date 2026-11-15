/**
 * Fake streaming API used when NEXT_PUBLIC_MOCK=true.
 *
 * It speaks the same StreamEvent contract as `streamAnswer`, so the UI and
 * hooks can be developed without Qdrant, an LLM key, or the Python backend.
 */

import type { AnswerOut, AskRequest, SourceOut, StreamEvent } from "./types";

const MOCK_SOURCES: SourceOut[] = [
  {
    n: 1,
    title: "Sliding Window — Padho with Pratyush",
    start_label: "02:35",
    url: "https://youtu.be/M7lc1UVf-VE?t=155",
    domain: "dsa",
    score: 0.84,
  },
  {
    n: 2,
    title: "Longest Substring Without Repeats — Padho with Pratyush",
    start_label: "12:10",
    url: "https://youtu.be/dQw4w9WgXcQ?t=730",
    domain: "dsa",
    score: 0.79,
  },
  {
    n: 3,
    title: "Sliding Window — Padho with Pratyush",
    start_label: "07:05",
    url: "https://youtu.be/M7lc1UVf-VE?t=425",
    domain: "dsa",
    score: 0.71,
  },
];

const ANSWER = [
  "A sliding window is a contiguous range of a sequence that expands and contracts as you scan it [1]. ",
  "It turns many nested-loop problems into a single pass, so the typical complexity drops from O(n²) to O(n) [2]. ",
  "Keep two pointers (`left`, `right`), move `right` to include a new element, then move `left` until the window is valid again [1][3]. ",
  "\n\nUse it for fixed-size windows (maximum sum of k elements) and variable-size windows (longest substring without repeats).",
].join("");

function chunkText(text: string, size: number): string[] {
  const words = text.split(/(\s+)/);
  const chunks: string[] = [];
  let current = "";
  for (const word of words) {
    current += word;
    if (current.length >= size) {
      chunks.push(current);
      current = "";
    }
  }
  if (current) chunks.push(current);
  return chunks;
}

function abortError(): DOMException {
  return new DOMException("The stream was cancelled.", "AbortError");
}

function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }

    const timer = window.setTimeout(() => {
      cleanup();
      resolve();
    }, ms);

    const onAbort = () => {
      window.clearTimeout(timer);
      cleanup();
      reject(abortError());
    };
    const cleanup = () => signal?.removeEventListener("abort", onAbort);

    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function makeAnswer(body: AskRequest, found: boolean): AnswerOut {
  return {
    query: body.query,
    answer: found
      ? ANSWER
      : "I couldn't find this in the indexed videos. Try asking about DSA or system design.",
    sources: found ? MOCK_SOURCES : [],
    found,
    uncited: !found,
    model: "mock/model",
    latency_ms: { total_ms: 420 },
    usage: { input_tokens: 120, output_tokens: 64 },
  };
}

export async function* mockStream(
  body: AskRequest,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  await delay(260, signal);

  const offTopic = /\b(bread|recipe|weather|cook|off.?topic)\b/i.test(
    body.query,
  );

  if (offTopic) {
    yield { type: "done", answer: makeAnswer(body, false) };
    return;
  }

  for (const chunk of chunkText(ANSWER, 7)) {
    await delay(45, signal);
    yield { type: "token", text: chunk };
  }

  await delay(120, signal);
  yield { type: "done", answer: makeAnswer(body, true) };
}
