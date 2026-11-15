/**
 * Citation tokenizer for answers such as "A window is a range [1][2]."
 *
 * Only numbers that match a known source become clickable chips. Unknown
 * numbers (for example `[99]`) stay as literal text instead of being dropped.
 */

export type CitationPart =
  | { type: "text"; text: string }
  | { type: "citation"; n: number };

const CITATION_PATTERN = /\[(\d{1,3})\]/g;

export function tokenizeCitations(
  text: string,
  validNumbers: Iterable<number>,
): CitationPart[] {
  const valid =
    validNumbers instanceof Set ? validNumbers : new Set(validNumbers);
  const parts: CitationPart[] = [];

  const pushText = (value: string) => {
    if (!value) return;
    const previous = parts[parts.length - 1];
    if (previous?.type === "text") previous.text += value;
    else parts.push({ type: "text", text: value });
  };

  let cursor = 0;
  let match: RegExpExecArray | null;
  CITATION_PATTERN.lastIndex = 0;

  while ((match = CITATION_PATTERN.exec(text)) !== null) {
    const n = Number(match[1]);
    if (!valid.has(n)) continue; // leave invalid markers in the text

    pushText(text.slice(cursor, match.index));
    parts.push({ type: "citation", n });
    cursor = match.index + match[0].length;
  }

  pushText(text.slice(cursor));
  return parts;
}

/** Unique, valid citation numbers in order of first appearance. */
export function citationNumbers(
  text: string,
  validNumbers: Iterable<number>,
): number[] {
  const valid =
    validNumbers instanceof Set ? validNumbers : new Set(validNumbers);
  const seen = new Set<number>();
  const result: number[] = [];
  CITATION_PATTERN.lastIndex = 0;

  for (const match of text.matchAll(CITATION_PATTERN)) {
    const n = Number(match[1]);
    if (valid.has(n) && !seen.has(n)) {
      seen.add(n);
      result.push(n);
    }
  }
  return result;
}
