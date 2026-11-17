"""Runner tests: per-question isolation, adapters, caching, selection, citations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import build_golden as bg
import run_eval as re
from judge import JudgeResult


def make_question(
    id_value: str = "q-001",
    *,
    domain: str = "dsa",
    category: str = "concept",
    answerable: bool = True,
    split: str = "dev",
    video: str = "video-a",
) -> bg.GoldenQuestion:
    return bg.GoldenQuestion(
        id=id_value,
        question="how does binary search work",
        domain=domain,
        category=category,
        answerable=answerable,
        gold_sources=[]
        if not answerable
        else [bg.GoldSource(video_id=video, start=0, end=100)],
        reference_answer="" if not answerable else "It halves the search space.",
        key_points=[] if not answerable else ["halves the range", "needs sorted input"],
        reviewed=True,
        index_snapshot={"total_chunks": 10},
        split=split,
    )


def retrieved_item(
    video: str, start: int, end: int, *, title: str = "Video"
) -> dict[str, object]:
    return {
        "rank": 1,
        "score": 0.9,
        "chunk_id": f"{video}_{start}",
        "video_id": video,
        "title": title,
        "channel": "channel",
        "domain": "dsa",
        "start": start,
        "end": end,
        "start_label": f"{start // 60}:{start % 60:02d}",
        "url": f"https://youtu.be/{video}?t={start}",
        "text": f"transcript text for {video} {start}",
        "snippet": "snippet",
    }


class FakePipeline:
    def __init__(self, output: re.PipelineOutput | None = None, error: Exception | None = None):
        self.output = output
        self.error = error
        self.calls = 0

    def run(self, question: bg.GoldenQuestion) -> re.PipelineOutput:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.output is not None
        return self.output


def base_output(**overrides: object) -> re.PipelineOutput:
    values: dict[str, object] = {
        "answer": "Binary search halves the range each step [1].",
        "sources": [
            {
                "n": 1,
                "title": "Binary Search",
                "start_label": "0:00",
                "url": "https://youtu.be/video-a?t=0",
                "domain": "dsa",
                "score": 0.9,
            }
        ],
        "found": True,
        "uncited": False,
        "model": "fake-generator",
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "retrieved": [
            retrieved_item("video-a", 0, 80),
            retrieved_item("video-x", 0, 80),
            retrieved_item("video-a", 90, 170),
        ],
        "context": "[1] context",
        "latency_ms": {"retrieval": 10.0, "generation": 20.0, "total": 31.0},
    }
    values.update(overrides)
    return re.PipelineOutput(**values)  # type: ignore[arg-type]


def test_evaluate_one_success_and_retrieval_metrics() -> None:
    pipeline = FakePipeline(base_output())
    record = re.evaluate_one(make_question(), pipeline, re.RunConfig(top_k=3), None)

    assert record["status"] == "ok"
    assert record["error"] is None
    assert record["abstained"] is False
    assert record["retrieval"]["n_gold"] == 1
    assert record["retrieval"]["n_retrieved"] == 3
    assert record["retrieval"]["n_relevant_retrieved"] == 2
    assert record["retrieval"]["precision"] == pytest.approx(2 / 3)
    assert record["retrieval"]["recall"] == 1.0
    assert record["usage"] == {"input_tokens": 100, "output_tokens": 20}


def test_one_failing_question_never_raises() -> None:
    pipeline = FakePipeline(error=RuntimeError("index exploded"))
    record = re.evaluate_one(make_question(), pipeline, re.RunConfig(), None)
    assert record["status"] == "error"
    assert "index exploded" in (record["error"] or "")
    assert record["retrieval"] is None


def test_false_abstention_is_flagged() -> None:
    output = base_output(
        answer="I couldn't find this in the indexed videos.",
        found=True,
        uncited=False,
        sources=[],
    )
    record = re.evaluate_one(
        make_question(), FakePipeline(output), re.RunConfig(), None
    )
    assert record["abstained"] is True
    assert record["answerable"] is True


def test_off_topic_abstention_has_no_retrieval_metrics() -> None:
    question = make_question(
        "off-001", category="off_topic", answerable=False
    )
    output = base_output(
        answer="No relevant content found in the indexed videos.",
        found=False,
        sources=[],
        retrieved=[],
        context="",
    )
    record = re.evaluate_one(question, FakePipeline(output), re.RunConfig(), None)
    assert record["abstained"] is True
    assert record["retrieval"] is None
    assert record["judge"] == {}


class StubJudge:
    provider = "stub"
    model = "stub-judge"

    def __init__(self) -> None:
        self.total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def _bump(self) -> None:
        self.total_usage["input_tokens"] += 10
        self.total_usage["output_tokens"] += 5
        self.total_usage["total_tokens"] += 15

    def faithfulness(self, question: str, answer: str, context: str) -> JudgeResult:
        self._bump()
        return JudgeResult(value=0.5, reason="1/2 supported")

    def answer_relevance(self, question: str, answer: str) -> JudgeResult:
        self._bump()
        return JudgeResult(value=0.75, reason="mostly relevant")

    def correctness(
        self,
        question: str,
        answer: str,
        reference_answer: str,
        key_points: list[str],
    ) -> JudgeResult:
        self._bump()
        return JudgeResult(value=0.8, reason="2/2 covered")

    def citation_accuracy(
        self, question: str, answer: str, citations: list[dict[str, object]]
    ) -> JudgeResult:
        self._bump()
        return JudgeResult(value=1.0, reason="1/1 citations supported")


def test_judge_is_wired_and_usage_delta_is_recorded() -> None:
    judge = StubJudge()
    record = re.evaluate_one(
        make_question(), FakePipeline(base_output()), re.RunConfig(), judge  # type: ignore[arg-type]
    )
    assert record["judge"]["faithfulness"]["value"] == 0.5
    assert record["judge"]["answer_relevance"]["value"] == 0.75
    assert record["judge"]["correctness"]["value"] == 0.8
    assert record["judge"]["citation_accuracy"]["value"] == 1.0
    assert record["judge_usage"] == {"input_tokens": 40, "output_tokens": 20}


def test_scoped_retriever_injects_settings() -> None:
    class BaseRetriever:
        last_timings = {"embed_ms": 1.0, "search_ms": 2.0}

        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def search(self, query: str, **kwargs: object) -> list[str]:
            self.kwargs = kwargs
            return ["result"]

    base = BaseRetriever()
    capture: dict[str, object] = {}
    config = re.RunConfig(
        top_k=4, threshold=0.55, use_mmr=False, use_reranker=True
    )
    scoped = re.ScopedRetriever(base, "dsa", config, capture)
    results = scoped.search("question", top_k=4)

    assert results == ["result"]
    assert base.kwargs["domain"] == "dsa"
    assert base.kwargs["score_threshold"] == 0.55
    assert base.kwargs["use_mmr"] is False
    assert base.kwargs["use_reranker"] is True
    assert capture["results"] == ["result"]
    assert scoped.last_timings["search_ms"] == 2.0


def test_config_hash_is_stable_and_sensitive() -> None:
    config = re.RunConfig(top_k=5, threshold=0.35)
    assert config.config_hash({}) == config.config_hash({})
    assert config.config_hash({}) != re.RunConfig(top_k=6, threshold=0.35).config_hash({})
    # The index snapshot is part of the key.
    assert config.config_hash({"total_chunks": 1}) != config.config_hash(
        {"total_chunks": 2}
    )


def test_select_questions_by_split_domain_and_limit() -> None:
    questions = [
        make_question("dev-dsa", split="dev"),
        make_question("test-dsa", split="test"),
        make_question("test-sd", domain="system_design", split="test"),
        make_question("off", category="off_topic", answerable=False, split="test"),
    ]
    config = re.RunConfig(split="test")
    assert [q.id for q in re.select_questions(questions, config)] == [
        "test-dsa",
        "test-sd",
        "off",
    ]
    domain_config = re.RunConfig(split="test", domain="system_design")
    assert [q.id for q in re.select_questions(questions, domain_config)] == ["test-sd"]
    limited = re.RunConfig(split="all", limit=2)
    assert [q.id for q in re.select_questions(questions, limited)] == [
        "dev-dsa",
        "test-dsa",
    ]


def test_cap_items_matches_context_budget() -> None:
    items = [
        {"title": "t", "start_label": "0:00", "url": "u", "text": "x" * 800}
        for _ in range(5)
    ]
    capped = re.cap_items(items, max_chars=2000)
    assert 0 < len(capped) < len(items)
    assert len(re.format_context(capped)) <= 2000


def test_build_citation_items_maps_markers_to_sources() -> None:
    kept = [
        retrieved_item("video-a", 0, 80, title="First"),
        retrieved_item("video-b", 0, 80, title="Second"),
        retrieved_item("video-c", 0, 80, title="Uncited"),
    ]
    answer = "The first fact is true [1]. The second fact is also true [2]."
    sources = [{"n": 1}, {"n": 2}]
    items = re.build_citation_items(answer, sources, kept)
    assert [item["n"] for item in items] == [1, 2]
    assert items[0]["source_title"] == "First"
    assert "[1]" in items[0]["sentence"]
    assert "Uncited" not in {item["source_title"] for item in items}


def test_estimate_skips_judge_when_disabled() -> None:
    questions = [make_question(), make_question("off", category="off_topic", answerable=False)]
    estimate = re.estimate_run(questions, re.RunConfig(no_judge=True))
    assert estimate["judge_calls"] == 0
    assert estimate["generator_calls"] == 2

    with_judge = re.estimate_run(questions, re.RunConfig())
    assert with_judge["judge_calls"] > 0


def test_cache_round_trip_only_returns_successes(tmp_path: Path) -> None:
    path = re.cache_path(tmp_path, "abc123", "dsa-001")
    record = {"id": "dsa-001", "status": "ok", "answer": "x"}
    re.save_cached(path, record)
    assert re.load_cached(path) == record

    error_path = re.cache_path(tmp_path, "abc123", "dsa-002")
    re.save_cached(error_path, {"id": "dsa-002", "status": "error"})
    assert re.load_cached(error_path) is None


def test_run_end_to_end_with_fake_pipeline_and_resume(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "dsa.jsonl").write_text(
        json.dumps({"video_id": "video-a", "start": 0, "end": 10}) + "\n",
        encoding="utf-8",
    )
    (chunks_dir / "system_design.jsonl").write_text("", encoding="utf-8")

    config = re.RunConfig(
        no_judge=True,
        sleep_seconds=0,
        split="all",
        chunks_dir=chunks_dir,
        results_dir=tmp_path / "results",
        cache_dir=tmp_path / "cache",
    )
    pipeline = FakePipeline(base_output())
    summary = re.run(
        config,
        pipeline=pipeline,
        judge=None,
        questions=[make_question()],
        confirm=False,
    )
    assert summary["n_questions"] == 1
    assert summary["n_errors"] == 0
    assert pipeline.calls == 1

    run_dirs = list((tmp_path / "results").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "summary.json").exists()
    assert (run_dirs[0] / "summary.md").exists()
    assert (run_dirs[0] / "failures.md").exists()
    assert (run_dirs[0] / "questions.jsonl").exists()

    # A resumed run with the same config hash reuses the cached question.
    resumed = re.RunConfig(**{**config.__dict__, "resume": True})
    pipeline_again = FakePipeline(base_output())
    re.run(
        resumed,
        pipeline=pipeline_again,
        judge=None,
        questions=[make_question()],
        confirm=False,
    )
    assert pipeline_again.calls == 0


def test_validate_gold_videos_against_chunk_files(tmp_path: Path) -> None:
    dsa = tmp_path / "dsa.jsonl"
    sd = tmp_path / "system_design.jsonl"
    dsa.write_text(
        json.dumps({"video_id": "video-a", "start": 0, "end": 10}) + "\n",
        encoding="utf-8",
    )
    sd.write_text(
        json.dumps({"video_id": "video-b", "start": 0, "end": 10}) + "\n",
        encoding="utf-8",
    )
    re.validate_gold_videos([make_question(video="video-a")], tmp_path)
    with pytest.raises(ValueError, match="not in the index"):
        re.validate_gold_videos([make_question(video="missing")], tmp_path)
