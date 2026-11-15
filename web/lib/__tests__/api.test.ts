import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, decodeEvent, streamAnswer } from "../api";
import type { AnswerOut, StreamEvent } from "../types";

function sseResponse(
  chunks: string[],
  status = 200,
  headers: Record<string, string> = {},
): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(stream, {
    status,
    headers: { "Content-Type": "text/event-stream", ...headers },
  });
}

const ANSWER: AnswerOut = {
  query: "q",
  answer: "A [1]",
  sources: [
    {
      n: 1,
      title: "Sliding Window",
      start_label: "00:10",
      url: "https://youtu.be/M7lc1UVf-VE?t=10",
      domain: "dsa",
      score: 0.9,
    },
  ],
  found: true,
  uncited: false,
  model: "test/model",
  latency_ms: { total_ms: 1 },
  usage: { input_tokens: 1, output_tokens: 1 },
};

async function collect(): Promise<StreamEvent[]> {
  const events: StreamEvent[] = [];
  for await (const event of streamAnswer({
    query: "q",
    domain: "dsa",
    top_k: 5,
    history: [],
  })) {
    events.push(event);
  }
  return events;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("streamAnswer", () => {
  it("decodes tokens and done across arbitrarily split chunks", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        sseResponse([
          'event: token\ndata: {"text":"A ',
          `[1]"}\n\nevent: done\ndata: ${JSON.stringify(ANSWER)}`,
          "\n\n",
        ]),
      ),
    );

    await expect(collect()).resolves.toEqual([
      { type: "token", text: "A [1]" },
      { type: "done", answer: ANSWER },
    ]);
  });

  it("turns a 429 envelope and Retry-After into a typed error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            error: {
              code: "rate_limited",
              message: "Too many requests.",
              request_id: "rid-1",
            },
          }),
          {
            status: 429,
            headers: {
              "Content-Type": "application/json",
              "Retry-After": "42",
            },
          },
        ),
      ),
    );

    const error = await collect().catch((cause: unknown) => cause);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 429,
      code: "rate_limited",
      requestId: "rid-1",
      retryAfterS: 42,
    });
  });

  it("surfaces a mid-stream error event", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          sseResponse([
            'event: token\ndata: {"text":"partial"}\n\n',
            'event: error\ndata: {"message":"LLM failed"}\n\n',
          ]),
        ),
    );

    await expect(collect()).resolves.toEqual([
      { type: "token", text: "partial" },
      { type: "error", message: "LLM failed" },
    ]);
  });
});

describe("decodeEvent", () => {
  it("reports an invalid done payload as a stream error", () => {
    expect(decodeEvent({ event: "done", data: "{}" })).toEqual({
      type: "error",
      message: "The server returned an invalid answer.",
    });
  });

  it("ignores the end event", () => {
    expect(decodeEvent({ event: "end", data: "null" })).toBeNull();
  });
});
