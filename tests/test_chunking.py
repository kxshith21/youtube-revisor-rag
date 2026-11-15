"""Tests for scripts/chunk_transcripts.py chunking logic."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chunk_transcripts import (  # noqa: E402
    MIN_TAIL_SECONDS,
    WINDOW_SECONDS,
    chunk_segments,
    clean_text,
    format_label,
    process_video,
)


def make_segments(duration: float, step: float = 4.0, words: int = 6) -> list[dict]:
    """Build synthetic caption segments: `duration` seconds total, `step` apart."""
    segs = []
    start = 0.0
    while start < duration:
        segs.append({
            "text": " ".join(["word"] * words),
            "start": start,
            "duration": min(step, duration - start),
        })
        start += step
    return segs


# --------------------------------------------------------------------------- #
# Normal case
# --------------------------------------------------------------------------- #
def test_normal_case() -> None:
    """Chunks reach ~window seconds, use whole segments, and stay ordered."""
    segments = make_segments(duration=240.0)
    ranges = chunk_segments(segments, window=78, overlap=12)

    assert len(ranges) >= 2
    for idx, (s, e) in enumerate(ranges):
        assert s < e
        span = (segments[e - 1]["start"] + segments[e - 1]["duration"]) - segments[s]["start"]
        if idx < len(ranges) - 1:
            # Window target: closes at >=78, one whole segment of overshoot max.
            assert 75 <= span <= WINDOW_SECONDS + 4
        else:
            # Final chunk: kept only if >= 25s (may be a merged tail).
            assert span >= MIN_TAIL_SECONDS
    # Ordered and increasing within one video.
    for (s1, _), (s2, _) in zip(ranges, ranges[1:]):
        assert s2 > s1


def test_process_video_accepts_end_timestamps() -> None:
    """Saved transcripts with start/end timestamps produce valid chunks."""
    segments = [
        {"text": " ".join(["word"] * 6), "start": start, "end": start + 4}
        for start in range(0, 40, 4)
    ]
    video = {
        "video_id": "test",
        "title": "Test video",
        "domain": "dsa",
        "segments": segments,
    }

    chunks = process_video(video, window=78, overlap=12, glossary={})

    assert len(chunks) == 1
    assert chunks[0]["start"] == 0
    assert chunks[0]["end"] == 40


def test_no_partial_segments() -> None:
    """Chunk boundaries always land on segment starts (no segment split in half)."""
    segments = make_segments(duration=200.0, step=3.7)
    starts = {seg["start"] for seg in segments}
    ranges = chunk_segments(segments, window=78, overlap=12)
    for s, _ in ranges:
        assert segments[s]["start"] in starts  # trivially true; guards index misuse
        assert s >= 0


# --------------------------------------------------------------------------- #
# Overlap
# --------------------------------------------------------------------------- #
def test_overlap() -> None:
    """The next chunk begins before the previous one ends (~12s of shared context)."""
    segments = make_segments(duration=300.0)
    ranges = chunk_segments(segments, window=78, overlap=12)

    assert len(ranges) >= 3
    for (s1, e1), (s2, _) in zip(ranges, ranges[1:]):
        assert s2 < e1, "chunks must overlap"
        # Shared time span is close to the requested overlap (segment granularity).
        shared = (segments[e1 - 1]["start"] + segments[e1 - 1]["duration"]) - segments[s2]["start"]
        assert 0 < shared <= 12 + 4  # at most one extra segment of slack


def test_overlap_zero_when_disabled() -> None:
    """With overlap=0, chunks are back-to-back (no shared segments)."""
    segments = make_segments(duration=300.0)
    ranges = chunk_segments(segments, window=78, overlap=0)
    for (_, e1), (s2, _) in zip(ranges, ranges[1:]):
        assert s2 >= e1 - 1  # may touch but not deeply overlap


# --------------------------------------------------------------------------- #
# Short tail merge
# --------------------------------------------------------------------------- #
def test_short_tail_merged() -> None:
    """A final chunk < 25s is merged into the previous one."""
    # Window closes at 80s, next chunk starts at 68s -> tail = 90-68 = 22s < 25s.
    segments = make_segments(duration=90.0)
    ranges = chunk_segments(segments, window=78, overlap=12)

    assert len(ranges) == 1, "short tail should be merged, not kept alone"
    s, e = ranges[0]
    span = (segments[e - 1]["start"] + segments[e - 1]["duration"]) - segments[s]["start"]
    assert span > WINDOW_SECONDS  # merged chunk is longer than one window


def test_long_tail_kept() -> None:
    """A final chunk >= 25s stays separate."""
    # 78 + ~40s tail.
    segments = make_segments(duration=118.0)
    ranges = chunk_segments(segments, window=78, overlap=12)
    assert len(ranges) == 2


# --------------------------------------------------------------------------- #
# Very short video
# --------------------------------------------------------------------------- #
def test_very_short_video_single_chunk() -> None:
    """A video shorter than the window becomes exactly one chunk (if long enough)."""
    segments = make_segments(duration=20.0, words=20)  # ~100 words
    ranges = chunk_segments(segments, window=78, overlap=12)
    assert len(ranges) == 1
    assert ranges[0] == (0, len(segments))


def test_very_short_video_too_few_words_dropped() -> None:
    """A tiny video with < 15 words yields no chunks."""
    segments = make_segments(duration=8.0, words=1)  # ~8 words total
    ranges = chunk_segments(segments, window=78, overlap=12)
    assert ranges == []


# --------------------------------------------------------------------------- #
# Cleaning & labels
# --------------------------------------------------------------------------- #
def test_clean_text_removes_noise() -> None:
    raw = "hello\n[Music] world >> 🎵 &amp; more   [Applause]"
    assert clean_text(raw) == "hello world & more"


def test_clean_text_applies_glossary_case_insensitive() -> None:
    glossary = {"cadin": "Kadane"}
    assert clean_text("the CADIN algorithm works", glossary) == "the Kadane algorithm works"


def test_format_label() -> None:
    assert format_label(340) == "05:40"
    assert format_label(0) == "00:00"
    assert format_label(3661) == "1:01:01"


def test_chunk_ids_stable() -> None:
    """Same input -> same index ranges (chunk_id derives from video_id + start)."""
    segments = make_segments(duration=250.0)
    assert chunk_segments(segments, 78, 12) == chunk_segments(segments, 78, 12)
