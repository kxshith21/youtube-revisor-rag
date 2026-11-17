"""Pure metric arithmetic tests: no network, no LLM, no files."""

from __future__ import annotations

import math

import metrics


def span(video: str, start: float, end: float) -> metrics.Span:
    return metrics.Span(video_id=video, start=start, end=end)


def test_overlap_and_relevance_boundary() -> None:
    gold = span("video-a", 100, 200)
    assert metrics.overlap_seconds(100, 200, 150, 250) == 50
    assert metrics.overlap_seconds(100, 200, 250, 300) == 0

    # Exactly 10 seconds of overlap counts by default.
    assert metrics.is_relevant(span("video-a", 190, 260), gold, min_overlap=10)
    # Just under 10 seconds does not.
    assert not metrics.is_relevant(span("video-a", 190.1, 260), gold, min_overlap=10)
    # Same time range but a different video is never relevant.
    assert not metrics.is_relevant(span("video-b", 100, 200), gold)


def test_retrieval_metrics_arithmetic() -> None:
    gold = [span("a", 0, 100), span("b", 200, 300)]
    retrieved = [span("a", 0, 80), span("x", 0, 80), span("b", 210, 300)]

    result = metrics.retrieval_metrics(retrieved, gold, k=3)
    assert result["n_gold"] == 2
    assert result["n_retrieved"] == 3
    assert result["n_relevant_retrieved"] == 2
    assert math.isclose(result["precision"], 2 / 3)
    assert math.isclose(result["recall"], 1.0)
    assert math.isclose(result["video_recall"], 1.0)
    assert math.isclose(result["hit"], 1.0)
    assert math.isclose(result["mrr"], 1.0)

    dcg = 1.0 + 0.0 + 1.0 / math.log2(4)
    idcg = 1.0 + 1.0 / math.log2(3)
    assert math.isclose(result["ndcg"], dcg / idcg)


def test_retrieval_metrics_no_results_and_no_gold() -> None:
    gold = [span("a", 0, 100)]
    empty = metrics.retrieval_metrics([], gold, k=5)
    assert empty["precision"] == 0.0
    assert empty["recall"] == 0.0
    assert empty["hit"] == 0.0
    assert empty["mrr"] == 0.0

    no_gold = metrics.retrieval_metrics([span("a", 0, 100)], [], k=5)
    assert no_gold["precision"] is None
    assert no_gold["n_gold"] == 0


def test_ndcg_perfect_single_rank() -> None:
    result = metrics.retrieval_metrics(
        [span("a", 0, 100)], [span("a", 0, 100)], k=1
    )
    assert result["ndcg"] == 1.0
    assert result["mrr"] == 1.0


def test_judged_precision() -> None:
    matched = [True, False, False]
    judged = [None, True, False]
    assert metrics.judged_precision(matched, judged) == 2 / 3
    assert metrics.judged_precision([], []) == 0.0


def test_bootstrap_ci_reproducible_and_bounded() -> None:
    values = [0.1, 0.4, 0.6, 0.9, 0.5, 0.3]
    low_a, high_a = metrics.bootstrap_ci(values, n_resamples=300, seed=7)
    low_b, high_b = metrics.bootstrap_ci(values, n_resamples=300, seed=7)
    assert (low_a, high_a) == (low_b, high_b)
    assert 0.1 <= low_a <= high_a <= 0.9

    low, high = metrics.bootstrap_ci([0.42], n_resamples=50)
    assert low == high == 0.42
    assert metrics.bootstrap_ci([], n_resamples=50) == (None, None)


def test_summarize_values_ignores_none() -> None:
    result = metrics.summarize_values([0.5, None, 1.0], n_resamples=100)
    assert result["n"] == 2
    assert math.isclose(result["value"], 0.75)


def test_abstention_detection_and_metrics() -> None:
    assert metrics.is_abstention("I couldn't find this in the indexed videos.", True)
    assert metrics.is_abstention("anything", found=False)
    assert not metrics.is_abstention("Binary search halves the range [1].", True)

    records = [
        {"answerable": False, "abstained": True},
        {"answerable": False, "abstained": False},
        {"answerable": True, "abstained": True},
        {"answerable": True, "abstained": False},
        {"answerable": True, "abstained": False},
    ]
    result = metrics.abstention_metrics(records)
    assert result["abstention_accuracy"] == 0.5
    assert math.isclose(result["false_abstention_rate"], 1 / 3)
    assert result["n_correct_abstentions"] == 1
    assert result["n_false_abstentions"] == 1

    assert metrics.abstention_metrics([{"answerable": True, "abstained": False}])[
        "abstention_accuracy"
    ] is None


def test_faithfulness_arithmetic() -> None:
    claims = [
        {"verdict": "supported"},
        {"verdict": "not_found"},
        {"verdict": "contradicted"},
    ]
    assert math.isclose(metrics.faithfulness_score(claims), 1 / 3)
    assert metrics.has_unsupported_claim(claims)
    assert metrics.faithfulness_score([]) is None
    assert not metrics.has_unsupported_claim([{"verdict": "supported"}])


def test_correctness_penalty_and_citation_score() -> None:
    assert math.isclose(
        metrics.correctness_score(["covered", "partial", "missing"]), 0.5
    )
    assert math.isclose(
        metrics.correctness_score(["covered"], ["wrong"]), 0.75
    )
    # Contradictions are capped so a contradictory answer scores 0 here.
    assert metrics.correctness_score(["covered"], ["a", "b", "c"]) == 0.5
    assert metrics.correctness_score([]) is None

    verdicts = [{"supported": True}, {"supported": False}]
    assert metrics.citation_score(verdicts) == 0.5
    assert metrics.citation_score([]) is None


def test_relevance_normalization() -> None:
    assert metrics.normalize_relevance(1) == 0.0
    assert metrics.normalize_relevance(3) == 0.5
    assert metrics.normalize_relevance(5) == 1.0
    assert metrics.normalize_relevance(0) == 0.0
    assert metrics.normalize_relevance(9) == 1.0


def test_percentile_and_safe_median() -> None:
    assert metrics.percentile([], 50) is None
    assert metrics.percentile([1, 2, 3], 50) == 2
    assert metrics.safe_median([]) == 0.0
    assert metrics.safe_median([3, 1, 2]) == 2
