"""Report tests: aggregation, CIs, grouped tables, statuses, comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

import report


def make_record(
    id_value: str,
    *,
    domain: str = "dsa",
    category: str = "concept",
    answerable: bool = True,
    status: str = "ok",
    abstained: bool = False,
    precision: float = 0.5,
    recall: float = 1.0,
    hit: float = 1.0,
    mrr: float = 0.5,
    ndcg: float = 0.5,
    gold_chunks: int = 1,
    faithfulness: float = 0.8,
    relevance: float = 0.9,
    correctness: float = 0.7,
    citation: float = 0.5,
    total_ms: float = 100.0,
    error: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "id": id_value,
        "question": f"question for {id_value}",
        "domain": domain,
        "category": category,
        "answerable": answerable,
        "split": "dev",
        "gold_sources": []
        if not answerable
        else [
            {"video_id": f"video-{index}", "start": 0, "end": 100}
            for index in range(gold_chunks)
        ],
        "reference_answer": "" if not answerable else "reference",
        "key_points": [] if not answerable else ["a", "b"],
        "status": status,
        "error": error,
        "answer": "refusal" if abstained else "an answer [1]",
        "found": not abstained,
        "abstained": abstained,
        "uncited": False,
        "model": "fake",
        "retrieved": []
        if not answerable
        else [
            {
                "video_id": "video-a",
                "start": 0,
                "end": 80,
                "title": "Video A",
            },
            {
                "video_id": "video-x",
                "start": 0,
                "end": 80,
                "title": "Video X",
            },
        ],
        "context": "",
        "sources": [{"n": 1, "title": "Video A"}],
        "latency_ms": {"retrieval": 50.0, "generation": 50.0, "total": total_ms},
        "usage": {
            "input_tokens": 10 if answerable else 0,
            "output_tokens": 5 if answerable else 0,
        },
        "judge_usage": {
            "input_tokens": 5 if answerable else 0,
            "output_tokens": 2 if answerable else 0,
        },
        "retrieval": None
        if not answerable
        else {
            "n_gold": gold_chunks,
            "n_retrieved": 2,
            "precision": precision,
            "recall": recall,
            "video_recall": recall,
            "hit": hit,
            "mrr": mrr,
            "ndcg": ndcg,
        },
        "judge": {}
        if not answerable
        else {
            "faithfulness": {"value": faithfulness, "reason": "faith reason", "details": {}},
            "answer_relevance": {"value": relevance, "reason": "rel reason", "details": {}},
            "correctness": {"value": correctness, "reason": "corr reason", "details": {}},
            "citation_accuracy": {"value": citation, "reason": "cite reason", "details": {}},
        },
    }
    return record


def test_build_summary_overall_and_groups() -> None:
    records = [
        make_record("dsa-1", precision=0.5),
        make_record("dsa-2", precision=0.6),
        make_record("sd-1", domain="system_design", precision=0.7),
        make_record(
            "off-1",
            category="off_topic",
            answerable=False,
            abstained=True,
        ),
    ]
    summary = report.build_summary(
        records, {"config": {"config_name": "unit", "top_k": 5}}
    )

    assert summary["n_questions"] == 4
    assert summary["n_answerable"] == 3
    assert summary["n_unanswerable"] == 1
    assert summary["overall"]["retrieval"]["precision"]["n"] == 3
    assert summary["overall"]["retrieval"]["precision"]["value"] == 0.6
    assert summary["overall"]["generation"]["faithfulness"]["value"] == pytest.approx(0.8)
    assert summary["overall"]["abstention"]["abstention_accuracy"]["value"] == 1.0
    assert summary["overall"]["abstention"]["false_abstention_rate"]["value"] == 0.0

    # One gold chunk at top_k=5 caps Precision@5 at 20% on every question.
    assert summary["overall"]["precision_ceiling"]["value"] == pytest.approx(0.2)
    assert summary["overall"]["precision_ceiling"]["top_k"] == 5

    # Off-topic uses the dsa placeholder in the schema but is excluded from
    # per-domain tables, so dsa only contains the two concept questions.
    assert summary["by_domain"]["dsa"]["n_questions"] == 2
    assert summary["by_domain"]["system_design"]["n_questions"] == 1
    assert summary["by_category"]["off_topic"]["n_questions"] == 1
    assert summary["overall"]["usage"]["generator"]["total_tokens"] == 45
    assert summary["overall"]["usage"]["judge"]["total_tokens"] == 21


def test_errors_are_not_counted_as_unanswerable() -> None:
    records = [make_record("err", status="error", error="boom")]
    summary = report.build_summary(records)
    assert summary["n_errors"] == 1
    assert summary["n_answerable"] == 0
    assert summary["n_unanswerable"] == 0


def test_precision_ceiling_is_mean_of_per_question_caps() -> None:
    records = [
        make_record("q1", gold_chunks=1),
        make_record("q2", gold_chunks=2),
    ]
    summary = report.build_summary(records, {"config": {"top_k": 5}})
    ceiling = summary["overall"]["precision_ceiling"]
    # min(1, 1/5) = 0.2 and min(1, 2/5) = 0.4; mean = 0.3.
    assert ceiling["value"] == pytest.approx(0.3)
    assert ceiling["n"] == 2
    assert ceiling["top_k"] == 5


def test_metric_labels_replace_k_with_the_run_top_k() -> None:
    assert report.label("precision", 3) == "Precision@3"
    assert report.label("precision", 10) == "Precision@10"
    assert report.label("hit", 5) == "Hit Rate@5"
    assert report.label("ndcg", 5) == "Normalized Discounted Cumulative Gain@5"
    assert report.label("faithfulness", 5) == "Faithfulness"
    assert report.label("false_abstention_rate") == "False Refusal Rate"
    assert report.label("precision") == "Precision@k"


def test_percentage_and_ci_formatting() -> None:
    assert report.fmt_pct(0.625) == "62.5%"
    assert report.fmt_pct(1.0) == "100.0%"
    assert report.fmt_pct(None) == "—"
    assert report.fmt_ci_pct({"ci_low": 0.625, "ci_high": 1.0}) == "62.5 - 100.0%"
    assert report.fmt_ci_pct({"ci_low": None, "ci_high": None}) == "—"


def test_status_thresholds_higher_is_better() -> None:
    assert report.status_key(0.85, "recall") == "good"
    assert report.status_key(0.849, "recall") == "ok"
    assert report.status_key(0.70, "recall") == "ok"
    assert report.status_key(0.699, "recall") == "needs_work"
    assert report.status_key(None, "recall") == "na"


def test_status_thresholds_lower_is_better() -> None:
    assert report.status_key(0.0, "false_abstention_rate") == "good"
    assert report.status_key(0.05, "false_abstention_rate") == "good"
    assert report.status_key(0.051, "false_abstention_rate") == "ok"
    assert report.status_key(0.15, "false_abstention_rate") == "ok"
    assert report.status_key(0.151, "false_abstention_rate") == "needs_work"


def test_precision_status_uses_the_attainable_ceiling() -> None:
    # 17% of all chunks is 85% of the 20% ceiling at top_k=5.
    assert report.status_key(0.17, "precision", ceiling=0.20) == "good"
    assert report.status_key(0.169, "precision", ceiling=0.20) == "ok"
    assert report.status_key(0.14, "precision", ceiling=0.20) == "ok"
    assert report.status_key(0.139, "precision", ceiling=0.20) == "needs_work"
    assert report.status_label(0.17, "precision", ceiling=0.20) == "✅ Good"


def test_small_sample_warning_boundary() -> None:
    assert (
        report.small_sample_warning(8)
        == "Small sample (n=8): treat these scores as a rough indication."
    )
    assert report.small_sample_warning(29).startswith("Small sample (n=29)")
    assert report.small_sample_warning(30) == ""
    assert report.small_sample_warning(0) == ""


def test_weakest_metric_and_readout() -> None:
    records = [make_record("q1", citation=0.1, precision=0.9)]
    summary = report.build_summary(records, {"config": {"top_k": 5}})
    assert summary["weakest"]["metric"] == "citation_accuracy"
    assert "Weakest area" in summary["readout"]
    assert "Citation Accuracy" in summary["readout"]
    assert "10.0%" in summary["readout"]


def test_all_targets_met_has_no_weakest() -> None:
    records = [
        make_record(
            "q1",
            precision=1.0,
            recall=1.0,
            hit=1.0,
            mrr=1.0,
            ndcg=1.0,
            faithfulness=1.0,
            relevance=1.0,
            correctness=1.0,
            citation=1.0,
        )
    ]
    summary = report.build_summary(records, {"config": {"top_k": 5}})
    assert summary["weakest"] is None
    assert "No ranked metric is below" in summary["readout"]


def test_precision_is_not_picked_as_the_weakest_area() -> None:
    records = [
        make_record(
            "q1",
            precision=0.0,
            recall=1.0,
            hit=1.0,
            mrr=1.0,
            ndcg=1.0,
            faithfulness=1.0,
            relevance=1.0,
            correctness=1.0,
            citation=1.0,
        )
    ]
    summary = report.build_summary(records, {"config": {"top_k": 5}})
    assert summary["weakest"] is None
    assert "Precision@5" in summary["readout"]
    assert "structurally capped" in summary["readout"]


def test_write_reports_creates_three_files(tmp_path: Path) -> None:
    records = [
        make_record("bad", precision=0.0, recall=0.0, citation=0.0),
        make_record("good", precision=0.9, recall=1.0, citation=0.9),
    ]
    meta = {
        "config": {"config_name": "unit-test", "top_k": 5},
        "config_hash": "abc123",
        "created_at": "2026-11-16T00:00:00+00:00",
    }
    summary = report.write_reports(records, meta, tmp_path)

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "failures.md").exists()
    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "# RAG evaluation" in markdown
    assert "## Headline" in markdown
    assert "## Run info" in markdown
    assert "## Overall" in markdown
    assert "## By domain" in markdown
    assert "## By category" in markdown
    assert "## Glossary" in markdown
    assert summary["n_questions"] == 2

    failures = (tmp_path / "failures.md").read_text(encoding="utf-8")
    assert "Run errors (0)" in failures
    assert "Precision@5 — 10 worst" in failures
    # The worst question is listed before the good one.
    assert failures.index("bad") < failures.index("good")


def test_markdown_uses_full_names_statuses_and_percentages() -> None:
    records = [make_record("q1", precision=0.17, citation=0.6)]
    summary = report.build_summary(
        records,
        {
            "config": {"config_name": "unit", "top_k": 5},
            "created_at": "2026-11-16T00:00:00+00:00",
        },
    )
    text = report.render_markdown(summary)

    assert "| Metric | Score | 95% CI | n | What it measures | Status |" in text
    assert "### Retrieval" in text
    assert "### Generation" in text
    assert "### Abstention" in text
    assert "### Performance" in text
    assert "Precision@5" in text
    assert "Mean Reciprocal Rank" in text
    assert "Normalized Discounted Cumulative Gain@5" in text
    assert "False Refusal Rate" in text
    assert "Abstention Accuracy" in text
    assert "17.0%" in text
    assert "Precision ceiling" in text
    assert "Small sample (n=1): treat these scores as a rough indication." in text
    assert "P@k" not in text
    assert "R@k" not in text
    assert "| Faith |" not in text

    precision_line = next(
        line for line in text.splitlines() if line.startswith("| Precision@5 |")
    )
    assert "✅ Good" in precision_line


def test_abstention_rows_are_always_visible() -> None:
    records = [make_record("q1")]
    summary = report.build_summary(records, {"config": {"top_k": 5}})
    text = report.render_markdown(summary)
    assert "Abstention Accuracy" in text
    assert "False Refusal Rate" in text
    assert "No unanswerable or off-topic questions" in text

    only_off_topic = report.build_summary(
        [
            make_record(
                "off-1",
                category="off_topic",
                answerable=False,
                abstained=True,
            )
        ],
        {"config": {"top_k": 5}},
    )
    text2 = report.render_markdown(only_off_topic)
    assert "False Refusal Rate" in text2
    assert "No answerable questions" in text2


def test_console_prints_grouped_table_without_abbreviations() -> None:
    records = [make_record("q1", precision=0.17)]
    summary = report.build_summary(
        records, {"config": {"config_name": "unit", "top_k": 5}}
    )
    text = report.render_console(summary)

    assert "RAG evaluation — unit" in text
    for heading in ("Retrieval", "Generation", "Abstention", "Performance"):
        assert heading in text
    assert "Metric" in text
    assert "What it measures" in text
    assert "95% CI" in text
    assert "Precision@5" in text
    assert "Mean Reciprocal Rank" in text
    assert "False Refusal Rate" in text
    assert "P@k" not in text
    assert "R@k" not in text
    assert "|" not in text  # console output is aligned text, not Markdown


def test_console_columns_are_aligned() -> None:
    table = report.console_table(
        ["Metric", "Score"], [["Precision@5", "20.0%"], ["Recall@5", "100.0%"]]
    )
    lines = table.splitlines()
    assert lines[0].index("Score") == lines[2].index("20.0%")
    assert lines[2].index("20.0%") == lines[3].index("100.0%")


def test_failures_lists_errors_and_abstention_misses(tmp_path: Path) -> None:
    records = [
        make_record("error-1", status="error", error="kaboom"),
        make_record("false-abstain", abstained=True),
        make_record("missed-abstain", answerable=False, abstained=False),
    ]
    summary = report.build_summary(records)
    text = report.render_failures(records, summary)

    assert "Run errors (1)" in text
    assert "False abstentions (1)" in text
    assert "Missed abstentions (1)" in text
    assert "kaboom" in text


def test_markdown_handles_missing_generation_metrics() -> None:
    records = [make_record("q1", answerable=False, abstained=True)]
    summary = report.build_summary(records, {"config": {"top_k": 5}})
    text = report.render_markdown(summary)
    assert "Generation metrics were not measured in this run" in text
    assert "### Performance" in text
    assert "## Glossary" in text


def test_compare_summaries_marks_better_and_worse() -> None:
    old = report.build_summary(
        [make_record("q1", precision=0.5, citation=0.9)],
        {"config": {"config_name": "old", "top_k": 5}},
    )
    new = report.build_summary(
        [make_record("q1", precision=0.7, citation=0.1)],
        {"config": {"config_name": "new", "top_k": 5}},
    )
    text = report.compare_summaries(old, new)
    assert "# Comparison: old → new" in text
    assert "Precision@5" in text
    assert "+20.0 pp" in text
    assert "better" in text  # precision improved
    assert "worse" in text  # citation accuracy regressed

    # Make the false refusal rate worse in the new run.
    old_abstain = report.build_summary([make_record("q1", abstained=False)])
    new_abstain = report.build_summary([make_record("q1", abstained=True)])
    text2 = report.compare_summaries(old_abstain, new_abstain)
    assert "False Refusal Rate" in text2
    assert "worse" in text2
