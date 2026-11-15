"""Tests for scripts/retrieve.py - no live Qdrant, no model downloads."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from retrieve import (  # noqa: E402
    MAX_QUERY_CHARS,
    SearchResult,
    apply_threshold,
    build_context,
    clean_query,
    merge_overlaps,
    mmr_select,
)


def R(
    start: int,
    end: int,
    score: float,
    video: str = "vidA",
    text: str = "some chunk text",
    chunk_id: str | None = None,
) -> SearchResult:
    """Convenience builder for test results."""
    return SearchResult(
        rank=0,
        score=score,
        chunk_id=chunk_id or f"{video}_{start}",
        video_id=video,
        title=f"Title of {video}",
        channel="chan",
        domain="dsa",
        start=start,
        end=end,
        start_label=f"{start // 60:02d}:{start % 60:02d}",
        url=f"https://youtu.be/{video}?t={start}",
        text=text,
        snippet=text[:250],
    )


# --------------------------------------------------------------------------- #
# Query cleaning
# --------------------------------------------------------------------------- #
def test_clean_query_strips_and_collapses() -> None:
    assert clean_query("  what \n is   sliding\twindow  ") == "what is sliding window"


def test_clean_query_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        clean_query("")
    with pytest.raises(ValueError, match="empty"):
        clean_query("   \n\t  ")


def test_clean_query_truncates_to_300() -> None:
    q = clean_query("x" * 500)
    assert len(q) == MAX_QUERY_CHARS


def test_clean_query_applies_glossary() -> None:
    glossary = {"cadin": "Kadane"}
    assert clean_query("explain the CADIN trick", glossary) == "explain the Kadane trick"


# --------------------------------------------------------------------------- #
# Threshold filtering
# --------------------------------------------------------------------------- #
def test_threshold_filtering() -> None:
    pairs = [(R(0, 80, 0.80), [1.0]), (R(100, 180, 0.30), [1.0]), (R(200, 280, 0.40), [1.0])]
    kept = apply_threshold(pairs, 0.35)
    assert [r.score for r, _ in kept] == [0.80, 0.40]


def test_threshold_empty_result_is_not_an_error() -> None:
    pairs = [(R(0, 80, 0.20), [1.0]), (R(90, 170, 0.10), [1.0])]
    assert apply_threshold(pairs, 0.35) == []


# --------------------------------------------------------------------------- #
# Overlap merging
# --------------------------------------------------------------------------- #
def test_merge_overlaps_same_video() -> None:
    """Adjacent same-video hits (within 20s) merge; higher score wins."""
    a = R(100, 180, 0.80, video="v1")
    b = R(175, 250, 0.70, video="v1")  # overlaps a (175 < 180)
    merged = merge_overlaps([(a, []), (b, [])], gap=20)
    assert len(merged) == 1
    r = merged[0][0]
    assert r.score == 0.80            # higher-scoring chunk kept as base
    assert r.start == 100 and r.end == 250  # range merged (min start, max end)
    assert r.url.endswith("?t=100")


def test_merge_keeps_separate_when_far_apart() -> None:
    a = R(0, 80, 0.9, video="v1")
    b = R(200, 280, 0.8, video="v1")  # 120s gap > 20s
    assert len(merge_overlaps([(a, []), (b, [])])) == 2


def test_merge_never_crosses_videos() -> None:
    a = R(100, 180, 0.9, video="v1")
    b = R(175, 250, 0.8, video="v2")  # same times, different video
    assert len(merge_overlaps([(a, []), (b, [])])) == 2


def test_merge_keeps_higher_score_even_if_later() -> None:
    a = R(100, 180, 0.60, video="v1")
    b = R(170, 260, 0.75, video="v1")  # later chunk scores higher
    merged = merge_overlaps([(a, []), (b, [])])
    assert merged[0][0].score == 0.75
    assert merged[0][0].start == 100 and merged[0][0].end == 260


# --------------------------------------------------------------------------- #
# MMR diversity
# --------------------------------------------------------------------------- #
def test_mmr_picks_different_videos() -> None:
    """Raw top scores are all video A; MMR must pull in video B."""
    v_a = [1.0, 0.0, 0.0]
    v_b = [0.0, 1.0, 0.0]  # orthogonal to A
    pairs = [
        (R(0, 80, 0.90, video="A"), list(v_a)),
        (R(90, 170, 0.89, video="A"), list(v_a)),
        (R(180, 260, 0.88, video="A"), list(v_a)),
        (R(0, 80, 0.80, video="B"), list(v_b)),
    ]
    picked = mmr_select(pairs, top_k=2, lam=0.7)
    videos = [r.video_id for r, _ in picked]
    assert videos[0] == "A"          # best relevance first
    assert videos[1] == "B"          # diversity beats a near-duplicate of A


def test_mmr_respects_top_k() -> None:
    pairs = [(R(i * 90, i * 90 + 80, 0.9 - i * 0.01, video=f"v{i}"), [1.0, 0.0])
             for i in range(6)]
    assert len(mmr_select(pairs, top_k=4)) == 4


# --------------------------------------------------------------------------- #
# Context builder
# --------------------------------------------------------------------------- #
def test_build_context_format() -> None:
    r1 = R(340, 418, 0.82, video="vX", text="first text")
    r1.title, r1.start_label, r1.url = "Sliding Window", "05:40", "https://youtu.be/vX?t=340"
    r2 = R(60, 140, 0.70, video="vY", text="second text")
    r2.title, r2.start_label, r2.url = "Two Pointers", "01:00", "https://youtu.be/vY?t=60"

    ctx = build_context([r1, r2])
    assert ctx == (
        "[1] Sliding Window @ 05:40 (https://youtu.be/vX?t=340)\nfirst text\n\n"
        "[2] Two Pointers @ 01:00 (https://youtu.be/vY?t=60)\nsecond text"
    )


def test_build_context_empty() -> None:
    assert build_context([]) == ""


# --------------------------------------------------------------------------- #
# SearchResult
# --------------------------------------------------------------------------- #
def test_from_point_snippet_and_payload() -> None:
    point = SimpleNamespace(
        score=0.82,
        payload={
            "chunk_id": "v1_340", "video_id": "v1", "title": "T", "channel": "C",
            "domain": "dsa", "start": 340, "end": 418, "start_label": "05:40",
            "url": "https://youtu.be/v1?t=340", "text": "y" * 300,
        },
        vector=[0.1, 0.2],
    )
    r = SearchResult.from_point(point, rank=1)
    assert r.rank == 1 and r.score == 0.82
    assert r.snippet == "y" * 250       # first 250 chars only
    assert len(r.to_dict()) == 13       # full UI schema present
