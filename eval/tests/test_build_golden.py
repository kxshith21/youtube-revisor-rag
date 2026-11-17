"""Golden dataset builder tests: sampling, CSV round trip, validation, splits."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from pydantic import ValidationError

import build_golden as bg


def make_chunk(
    video: str,
    index: int,
    *,
    words: int = 80,
    domain: str = "dsa",
    start: int | None = None,
    end: int | None = None,
) -> bg.Chunk:
    start = index * 80 if start is None else start
    end = start + 80 if end is None else end
    text = " ".join(f"token{index}_{position}" for position in range(words))
    return bg.Chunk(
        chunk_id=f"{video}_{start}",
        video_id=video,
        title=f"Video {video}",
        domain=domain,
        start=start,
        end=end,
        text=text,
        chunk_index=index,
    )


def make_question(
    id_value: str,
    *,
    domain: str = "dsa",
    category: str = "concept",
    answerable: bool = True,
    video: str = "video-a",
    start: int = 0,
    end: int = 100,
    split: str = "dev",
    reviewed: bool = True,
) -> bg.GoldenQuestion:
    return bg.GoldenQuestion(
        id=id_value,
        question=f"what does question {id_value} ask about binary search",
        domain=domain,
        category=category,
        answerable=answerable,
        gold_sources=[]
        if not answerable
        else [bg.GoldSource(video_id=video, start=start, end=end)],
        reference_answer="" if not answerable else "A short reference answer.",
        key_points=[] if not answerable else ["first point", "second point"],
        reviewed=reviewed,
        index_snapshot={"total_chunks": 10},
        split=split,
    )


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def test_sample_chunks_reproducible_and_capped_per_video() -> None:
    chunks = [
        make_chunk(video, index)
        for video in ("a", "b", "c")
        for index in range(6)
    ]
    first = bg.sample_chunks(chunks, 6, seed=42)
    second = bg.sample_chunks(chunks, 6, seed=42)
    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]
    assert len(first) == 6
    counts: dict[str, int] = {}
    for chunk in first:
        counts[chunk.video_id] = counts.get(chunk.video_id, 0) + 1
    assert max(counts.values()) <= bg.MAX_PER_VIDEO


def test_sample_chunks_skips_short_and_respects_capacity() -> None:
    chunks = [
        make_chunk("a", 0),
        make_chunk("a", 1, words=10),  # too short
        make_chunk("b", 0),
    ]
    sampled = bg.sample_chunks(chunks, 10, seed=1, min_words=60)
    assert {chunk.chunk_id for chunk in sampled} == {"a_0", "b_0"}
    assert bg.sample_chunks(chunks, 0, seed=1) == []


def test_neighbors_and_gold_source_cover_context() -> None:
    chunks = [make_chunk("a", index) for index in range(4)]
    by_video = bg.video_index(chunks)
    neighbors = bg.neighbors_for(chunks[1], by_video, before=1, after=1)
    assert [chunk.chunk_index for chunk in neighbors] == [0, 1, 2]
    gold = bg.gold_source_for(neighbors, chunks[1])
    assert gold.video_id == "a"
    assert gold.start == 0
    assert gold.end == 240


def test_keyword_phrase_and_typos_are_deterministic() -> None:
    phrase = bg.keyword_phrase("what is the difference between BFS and DFS")
    assert phrase and "what" not in phrase.split()
    assert phrase == bg.keyword_phrase("what is the difference between BFS and DFS")

    question = "explain the binary search algorithm clearly"
    first = bg.introduce_typos(question, random.Random(5), count=1)
    second = bg.introduce_typos(question, random.Random(5), count=1)
    assert first == second
    assert isinstance(first, str) and first


# --------------------------------------------------------------------------- #
# review.csv
# --------------------------------------------------------------------------- #
def test_review_csv_round_trip_and_edits(tmp_path: Path) -> None:
    question = make_question("dsa-001", reviewed=False)
    csv_path = tmp_path / "review.csv"
    bg.write_review_csv(
        [question], [], {"video-a": "Binary Search Video"}, csv_path
    )

    rows = bg.read_review_csv(csv_path)
    assert len(rows) == 1
    assert set(rows[0]) == set(bg.REVIEW_FIELDS)
    assert rows[0]["id"] == "dsa-001"
    assert rows[0]["keep"] == ""
    assert "youtu.be/video-a" in rows[0]["gold_url"]

    edited = dict(rows[0])
    edited.update(
        {
            "keep": "yes",
            "question": "how does binary search halve the range",
            "reference_answer": "It halves the remaining search space each step.",
            "key_points": "halves the range | needs sorted input",
            "gold_url": "https://youtu.be/4L4MYtxGh9s?t=160",
            "notes": "checked against the transcript",
        }
    )
    updated = bg.apply_review_row(question, edited)
    assert updated.reviewed is True
    assert updated.question.startswith("how does")
    assert updated.key_points == ["halves the range", "needs sorted input"]
    assert updated.gold_sources[0].video_id == "4L4MYtxGh9s"
    assert updated.gold_sources[0].start == 160
    # The CSV only edits the start, so the original duration is preserved.
    original_duration = question.gold_sources[0].end - question.gold_sources[0].start
    assert (
        updated.gold_sources[0].end - updated.gold_sources[0].start
        == original_duration
    )


def test_review_keep_flags() -> None:
    assert bg._is_kept("yes") and bg._is_kept(" TRUE ") and bg._is_kept("1")
    assert bg._is_rejected("no") and bg._is_rejected(" Drop ")
    assert not bg._is_kept("")
    assert not bg._is_rejected("")


# --------------------------------------------------------------------------- #
# Schema and final validation
# --------------------------------------------------------------------------- #
def test_gold_source_range_and_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        bg.GoldSource(video_id="a", start=100, end=100)
    with pytest.raises(ValidationError):
        bg.GoldenQuestion.model_validate(
            {
                **make_question("x").model_dump(),
                "unexpected": "field",
            }
        )


def test_off_topic_and_multi_video_invariants() -> None:
    with pytest.raises(ValidationError):
        make_question("off", category="off_topic", answerable=True)
    with pytest.raises(ValidationError, match="key point"):
        bg.GoldenQuestion.model_validate(
            {**make_question("no-points").model_dump(), "key_points": []}
        )
    with pytest.raises(ValidationError):
        bg.GoldenQuestion(
            id="multi",
            question="compare two things from different videos",
            domain="dsa",
            category="multi_video",
            answerable=True,
            gold_sources=[bg.GoldSource(video_id="only-one", start=0, end=10)],
            reference_answer="reference",
            key_points=["a"],
        )


def test_validate_final_rejects_bad_data() -> None:
    known = {"video-a", "video-b"}
    good = [make_question(f"q{i}") for i in range(3)]
    bg.validate_final(good, known, min_answerable=1, require_reviewed=False)

    bad_video = [make_question("bad", video="missing")]
    with pytest.raises(ValueError, match="not in the index"):
        bg.validate_final(bad_video, known, min_answerable=1, require_reviewed=False)

    duplicate = [make_question("same"), make_question("same")]
    with pytest.raises(ValueError, match="Duplicate"):
        bg.validate_final(duplicate, known, min_answerable=1, require_reviewed=False)

    with pytest.raises(ValueError, match="at least 40"):
        bg.validate_final(good, known, min_answerable=40, require_reviewed=False)

    unreviewed = [make_question("u", reviewed=False)]
    with pytest.raises(ValueError, match="not been reviewed"):
        bg.validate_final(unreviewed, known, min_answerable=1, require_reviewed=True)


# --------------------------------------------------------------------------- #
# Splits and composition
# --------------------------------------------------------------------------- #
def test_assign_splits_is_reproducible_and_stratified() -> None:
    def build() -> list[bg.GoldenQuestion]:
        questions = [make_question(f"dsa-{i:03d}") for i in range(20)]
        questions += [
            make_question(f"off-{i:03d}", category="off_topic", answerable=False)
            for i in range(10)
        ]
        return questions

    first = bg.assign_splits(build(), ratio=0.7, seed=11)
    second = bg.assign_splits(build(), ratio=0.7, seed=11)
    assert {q.id: q.split for q in first} == {q.id: q.split for q in second}

    by_category: dict[str, set[str]] = {}
    for question in first:
        by_category.setdefault(question.category, set()).add(question.split)
    assert by_category["concept"] == {"dev", "test"}
    assert by_category["off_topic"] == {"dev", "test"}


def test_assign_splits_rejects_bad_ratio() -> None:
    with pytest.raises(ValueError):
        bg.assign_splits([], ratio=1.0)
    with pytest.raises(ValueError):
        bg.assign_splits([], ratio=0.0)


def test_composition_stats_counts() -> None:
    questions = [
        make_question("dsa-001"),
        make_question("dsa-002", category="typo"),
        make_question("sd-001", domain="system_design"),
        make_question("off-001", category="off_topic", answerable=False),
    ]
    stats = bg.composition_stats(questions)
    assert stats["total"] == 4
    assert stats["answerable"] == 3
    assert stats["unanswerable"] == 1
    assert stats["dsa_answerable"] == 2
    assert stats["sd_answerable"] == 1
    assert stats["off_topic"] == 1
    assert stats["typo_vague"] == 1
    assert stats["generated_share"] == 3 / 4
