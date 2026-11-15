import { describe, expect, it } from "vitest";
import { citationNumbers, tokenizeCitations } from "../citations";

describe("tokenizeCitations", () => {
  it("splits text around a citation", () => {
    expect(tokenizeCitations("See [1] for details.", [1])).toEqual([
      { type: "text", text: "See " },
      { type: "citation", n: 1 },
      { type: "text", text: " for details." },
    ]);
  });

  it("handles adjacent citations", () => {
    expect(tokenizeCitations("[1][2] and [3]", [1, 2, 3])).toEqual([
      { type: "citation", n: 1 },
      { type: "citation", n: 2 },
      { type: "text", text: " and " },
      { type: "citation", n: 3 },
    ]);
  });

  it("keeps invalid citation numbers as literal text", () => {
    expect(tokenizeCitations("See [9] and [1].", [1])).toEqual([
      { type: "text", text: "See [9] and " },
      { type: "citation", n: 1 },
      { type: "text", text: "." },
    ]);
  });

  it("returns one text part when nothing matches", () => {
    expect(tokenizeCitations("No citations here.", [1])).toEqual([
      { type: "text", text: "No citations here." },
    ]);
  });

  it("normalizes zero-padded numbers", () => {
    expect(tokenizeCitations("[01]", [1])).toEqual([
      { type: "citation", n: 1 },
    ]);
  });

  it("accepts an array of valid numbers", () => {
    expect(tokenizeCitations("see [2]", [1, 2])).toEqual([
      { type: "text", text: "see " },
      { type: "citation", n: 2 },
    ]);
  });
});

describe("citationNumbers", () => {
  it("returns unique valid numbers in order", () => {
    expect(citationNumbers("[2][1][2][9]", [1, 2])).toEqual([2, 1]);
  });

  it("returns an empty list without citations", () => {
    expect(citationNumbers("plain text", [1])).toEqual([]);
  });
});
