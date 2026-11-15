import { describe, expect, it } from "vitest";
import { flushSSE, parseSSEChunk } from "../sse";

describe("parseSSEChunk", () => {
  it("parses a complete event", () => {
    const result = parseSSEChunk(
      "",
      'event: token\ndata: {"text":"hi"}\n\n',
    );

    expect(result.events).toEqual([{ event: "token", data: '{"text":"hi"}' }]);
    expect(result.rest).toBe("");
  });

  it("buffers a chunk that splits the data and delimiter", () => {
    const first = parseSSEChunk(
      "",
      'event: token\ndata: {"text":"hel',
    );
    expect(first.events).toEqual([]);
    expect(first.rest).toBe('event: token\ndata: {"text":"hel');

    const second = parseSSEChunk(first.rest, 'lo"}\n\n');
    expect(second.events).toEqual([
      { event: "token", data: '{"text":"hello"}' },
    ]);
    expect(second.rest).toBe("");
  });

  it("parses multiple events in one chunk and keeps the remainder", () => {
    const result = parseSSEChunk(
      "",
      "event: a\ndata: 1\n\nevent: b\ndata: 2\n\nevent: c\ndata: 3",
    );

    expect(result.events).toEqual([
      { event: "a", data: "1" },
      { event: "b", data: "2" },
    ]);
    expect(result.rest).toBe("event: c\ndata: 3");
  });

  it("handles CRLF, comments, and heartbeat frames", () => {
    const result = parseSSEChunk(
      "",
      'event: token\r\ndata: {"text":"a"}\r\n\r\n: heartbeat\r\n\r\n',
    );

    expect(result.events).toEqual([
      { event: "token", data: '{"text":"a"}' },
    ]);
    expect(result.rest).toBe("");
  });

  it("joins multi-line data and defaults the event name", () => {
    const result = parseSSEChunk("", "data: line1\ndata: line2\n\n");
    expect(result.events).toEqual([
      { event: "message", data: "line1\nline2" },
    ]);
  });

  it("parses an error event", () => {
    const result = parseSSEChunk(
      "",
      'event: error\ndata: {"message":"boom"}\n\n',
    );

    expect(result.events).toEqual([
      { event: "error", data: '{"message":"boom"}' },
    ]);
  });
});

describe("flushSSE", () => {
  it("parses a final block without a trailing delimiter", () => {
    expect(flushSSE('event: done\ndata: {"ok":true}')).toEqual([
      { event: "done", data: '{"ok":true}' },
    ]);
  });

  it("ignores an empty buffer", () => {
    expect(flushSSE("")).toEqual([]);
    expect(flushSSE("\n\n")).toEqual([]);
  });
});
