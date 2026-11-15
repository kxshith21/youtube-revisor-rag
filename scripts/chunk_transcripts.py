"""Step 2 of the YouTube Reviser RAG pipeline.

Reads transcript JSONs from transcripts/<domain>/, splits them into time-based
overlapping chunks, and writes chunks/<domain>.jsonl.

Usage (from the project root):
    python scripts/chunk_transcripts.py --domain all --limit 5   # test run
    python scripts/chunk_transcripts.py --domain all             # full run
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
WINDOW_SECONDS = 78  # target chunk length (75-80s)
OVERLAP_SECONDS = 12  # context shared between adjacent chunks
MIN_WORDS = 15  # skip chunks with fewer words after cleaning
MIN_TAIL_SECONDS = 25  # merge a shorter final chunk into the previous one

ROOT = Path(__file__).resolve().parents[1]  # project root, CWD-independent
TRANSCRIPTS_DIR = ROOT / "transcripts"
CHUNKS_DIR = ROOT / "chunks"
GLOSSARY_PATH = ROOT / "glossary.json"

DOMAINS = ("dsa", "system_design")

# Caption noise: bracketed cues ([Music]...), ">>" separators, music/note symbols.
# NOTE: \U + 8 hex digits for astral-plane code points — \u only reads 4 digits.
NOISE_RE = re.compile(
    r"\[[^\]]*\]|>>+|[\u266a-\u266f\U0001d15e-\U0001d164\U0001f3b4-\U0001f3bb]+"
)


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def load_glossary() -> dict[str, str]:
    """Load wrong -> right term corrections; missing/broken file -> {} (no error)."""
    try:
        data = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def clean_text(text: str, glossary: dict[str, str] | None = None) -> str:
    """Remove caption noise, decode HTML entities, collapse whitespace,
    and apply case-insensitive glossary corrections."""
    text = html.unescape(text)          # &amp; -> &
    text = NOISE_RE.sub(" ", text)      # [Music], >>, note symbols
    text = text.replace("\n", " ").replace("\r", " ")
    if glossary:
        for wrong, right in glossary.items():
            text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def load_transcript(path: Path) -> dict[str, Any] | None:
    """Load one transcript JSON; returns None (with a warning) if empty/malformed."""
    try:
        if path.stat().st_size == 0:
            raise ValueError("file is empty")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data.get("segments"), list) or not data["segments"]:
            raise ValueError("missing or empty 'segments'")
        return data
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"  WARNING: skipping {path.name} ({exc})")
        return None


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _seg_start(seg: dict[str, Any]) -> float:
    return float(seg["start"])


def _seg_end(seg: dict[str, Any]) -> float:
    if "duration" in seg:
        return _seg_start(seg) + float(seg["duration"])
    return float(seg["end"])


def chunk_segments(
    segments: list[dict[str, Any]],
    window: int = WINDOW_SECONDS,
    overlap: int = OVERLAP_SECONDS,
) -> list[tuple[int, int]]:
    """Split one video's segments into (start_idx, end_idx) index ranges.

    - Whole segments only; a chunk closes once its span reaches `window`.
    - The next chunk starts ~`overlap` seconds before the previous chunk's end.
    - A final chunk shorter than MIN_TAIL_SECONDS is merged into the previous.
    - Ranges with fewer than MIN_WORDS (after cleaning) are dropped.
    Never crosses videos: call once per video.
    """
    n = len(segments)
    ranges: list[tuple[int, int]] = []

    i = 0
    while i < n:
        first_start = _seg_start(segments[i])
        last_end = _seg_end(segments[i])
        j = i + 1
        # Add whole segments until the chunk reaches the target window.
        while j < n and (last_end - first_start) < window:
            last_end = _seg_end(segments[j])
            j += 1
        ranges.append((i, j))

        if j >= n:
            break
        # Next chunk starts ~overlap seconds before this chunk's end.
        chunk_end = _seg_end(segments[j - 1])
        k = j - 1
        while k > i and (chunk_end - _seg_start(segments[k])) < overlap:
            k -= 1
        i = k if k > i else j  # always make progress

    # Merge a too-short final chunk into the previous one.
    if len(ranges) >= 2:
        tail_start, tail_end = ranges[-1]
        tail_dur = _seg_end(segments[tail_end - 1]) - _seg_start(segments[tail_start])
        if tail_dur < MIN_TAIL_SECONDS:
            ranges[-2] = (ranges[-2][0], ranges[-1][1])
            ranges.pop()

    # Drop chunks that are too thin after cleaning (< MIN_WORDS).
    kept: list[tuple[int, int]] = []
    for s, e in ranges:
        text = clean_text(" ".join(seg["text"] for seg in segments[s:e]))
        if len(text.split()) >= MIN_WORDS:
            kept.append((s, e))
    return kept


def format_label(seconds: float) -> str:
    """Format seconds as mm:ss (or h:mm:ss past an hour)."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- #
# Building & saving chunks
# --------------------------------------------------------------------------- #
def build_chunk(
    video: dict[str, Any],
    segments: list[dict[str, Any]],
    idx_range: tuple[int, int],
    chunk_index: int,
    glossary: dict[str, str],
) -> dict[str, Any]:
    """Assemble one output chunk with exact timestamps and stable id."""
    s, e = idx_range
    first, last = segments[s], segments[e - 1]
    start = int(_seg_start(first))
    end = int(_seg_end(last))
    video_id = video["video_id"]
    text = clean_text(" ".join(seg["text"] for seg in segments[s:e]), glossary)
    return {
        "chunk_id": f"{video_id}_{start}",
        "video_id": video_id,
        "title": video["title"],
        "channel": video.get("channel", ""),
        "domain": video["domain"],
        "chunk_index": chunk_index,
        "start": start,
        "end": end,
        "start_label": format_label(start),
        "url": f"https://youtu.be/{video_id}?t={start}",
        "text": text,
        "text_for_embedding": f"{video['title']} | {text}",
    }


def process_video(video: dict[str, Any], window: int, overlap: int,
                  glossary: dict[str, str]) -> list[dict[str, Any]]:
    """Chunk one video independently (chunks never cross videos)."""
    # Clean each segment's text up front (chunk word counts use cleaned text).
    segments = [
        {**seg, "text": clean_text(seg["text"], glossary)}
        for seg in video["segments"]
    ]
    ranges = chunk_segments(segments, window, overlap)
    return [build_chunk(video, segments, r, i, glossary)
            for i, r in enumerate(ranges)]


def validate_chunks(chunks: list[dict[str, Any]]) -> None:
    """Assert ordering, timestamp sanity, id uniqueness, and non-empty text.

    Note: chunks overlap by design, so within a video the check is strictly
    increasing starts (and ends), not "next start >= previous end".
    """
    seen_ids: set[str] = set()
    last_start: dict[str, int] = {}
    last_end: dict[str, int] = {}
    for c in chunks:
        assert c["start"] < c["end"], f"start >= end: {c['chunk_id']}"
        assert c["chunk_id"] not in seen_ids, f"duplicate id: {c['chunk_id']}"
        seen_ids.add(c["chunk_id"])
        assert c["text"].strip(), f"empty text: {c['chunk_id']}"
        vid = c["video_id"]
        if vid in last_start:  # chunks within a video must be ordered
            assert c["start"] > last_start[vid], f"start out of order: {c['chunk_id']}"
            assert c["end"] > last_end[vid], f"end out of order: {c['chunk_id']}"
        last_start[vid] = c["start"]
        last_end[vid] = c["end"]


def save_chunks(domain: str, chunks: list[dict[str, Any]]) -> Path:
    """Write chunks/<domain>.jsonl (utf-8, ensure_ascii=False, one per line)."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNKS_DIR / f"{domain}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return out_path


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_summary(domain: str, videos: int, chunks: list[dict[str, Any]]) -> None:
    """Per-domain stats: counts, average duration/words, shortest & longest."""
    if not chunks:
        print(f"  summary [{domain}]: {videos} videos, 0 chunks")
        return
    durations = [c["end"] - c["start"] for c in chunks]
    words = [len(c["text"].split()) for c in chunks]
    shortest = min(chunks, key=lambda c: c["end"] - c["start"])
    longest = max(chunks, key=lambda c: c["end"] - c["start"])
    print(f"  summary [{domain}]:")
    print(f"    videos processed: {videos}")
    print(f"    total chunks:     {len(chunks)}")
    print(f"    avg duration:     {sum(durations) / len(durations):.1f}s")
    print(f"    avg words/chunk:  {sum(words) / len(words):.1f}")
    print(f"    shortest chunk:   {shortest['chunk_id']} ({shortest['end'] - shortest['start']}s)")
    print(f"    longest chunk:    {longest['chunk_id']} ({longest['end'] - longest['start']}s)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk transcript JSONs into overlapping time-based chunks."
    )
    parser.add_argument("--domain", "-domain", choices=["dsa", "system_design", "all"],
                        default="all", help="which domain to process (default: all)")
    parser.add_argument("--window", type=int, default=WINDOW_SECONDS,
                        help=f"target chunk length in seconds (default: {WINDOW_SECONDS})")
    parser.add_argument("--overlap", type=int, default=OVERLAP_SECONDS,
                        help=f"overlap between chunks in seconds (default: {OVERLAP_SECONDS})")
    parser.add_argument("--limit", "-limit", type=int, default=None, metavar="N",
                        help="only process the first N videos per domain (for testing)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    domains = list(DOMAINS) if args.domain == "all" else [args.domain]
    glossary = load_glossary()
    if glossary:
        print(f"loaded glossary: {len(glossary)} correction(s)")

    for domain in domains:
        print(f"\n=== {domain} ===")
        files = sorted((TRANSCRIPTS_DIR / domain).glob("*.json"))
        if args.limit is not None:
            files = files[: args.limit]
        if not files:
            print("  no transcript files found")
            continue

        all_chunks: list[dict[str, Any]] = []
        processed = 0
        for path in files:
            video = load_transcript(path)
            if video is None:
                continue
            video.setdefault("domain", domain)
            processed += 1
            all_chunks.extend(process_video(video, args.window, args.overlap, glossary))

        validate_chunks(all_chunks)
        out_path = save_chunks(domain, all_chunks)
        print_summary(domain, processed, all_chunks)
        print(f"    written to:       {out_path.relative_to(ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
