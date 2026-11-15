import { describe, expect, it } from "vitest";
import { formatTimestamp } from "../time";

describe("formatTimestamp", () => {
  it.each([
    [0, "00:00"],
    [5, "00:05"],
    [65, "01:05"],
    [90.9, "01:30"],
    [3599, "59:59"],
    [3600, "1:00:00"],
    [3661, "1:01:01"],
    [7325, "2:02:05"],
  ])("formats %s seconds as %s", (seconds, expected) => {
    expect(formatTimestamp(seconds)).toBe(expected);
  });

  it("clamps invalid values", () => {
    expect(formatTimestamp(-10)).toBe("00:00");
    expect(formatTimestamp(Number.NaN)).toBe("00:00");
    expect(formatTimestamp(Number.POSITIVE_INFINITY)).toBe("00:00");
  });
});
