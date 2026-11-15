/**
 * Incremental Server-Sent Events parser.
 *
 * `fetch` streams arbitrary chunks, so an event can be split anywhere — even
 * in the middle of `\r\n\r\n`. Feed every chunk to `parseSSEChunk` and keep the
 * returned `rest` for the next call. `flushSSE` handles the final partial block
 * (for example when a proxy closes the stream without a trailing blank line).
 */

export interface SSERawEvent {
  event: string;
  data: string;
}

export interface ParseResult {
  events: SSERawEvent[];
  rest: string;
}

function parseBlock(block: string): SSERawEvent | null {
  const dataLines: string[] = [];
  let event = "message";

  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue; // blank line or heartbeat comment

    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }

  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

export function parseSSEChunk(buffer: string, chunk: string): ParseResult {
  // Normalising CR/CRLF keeps the delimiter search simple. A lone trailing
  // "\r" is harmless: when its "\n" arrives the pair still forms a delimiter.
  const combined = (buffer + chunk).replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const blocks = combined.split("\n\n");
  const rest = blocks.pop() ?? "";

  const events: SSERawEvent[] = [];
  for (const block of blocks) {
    const parsed = parseBlock(block);
    if (parsed) events.push(parsed);
  }
  return { events, rest };
}

/** Parse whatever is left in the buffer after the reader finishes. */
export function flushSSE(buffer: string): SSERawEvent[] {
  if (!buffer.trim()) return [];
  const normalised = buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const parsed = parseBlock(normalised);
  return parsed ? [parsed] : [];
}
