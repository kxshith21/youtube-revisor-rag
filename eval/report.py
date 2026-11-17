"""Turn per-question evaluation records into readable reports.

Writes three files into a run directory:

    summary.json   machine-readable numbers (means, 95% bootstrap CIs, groups)
    summary.md     human-readable report (overall, by domain, by category)
    failures.md    10 worst questions per metric, plus errors and abstention misses

Also supports comparison:

    python eval/report.py --compare eval/results/run-a eval/results/run-b

No network calls and no LLM calls happen here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

import metrics  # noqa: E402

RETRIEVAL_METRICS = (
    "precision",
    "recall",
    "video_recall",
    "hit",
    "mrr",
    "ndcg",
    "judged_precision",
)
GENERATION_METRICS = (
    "faithfulness",
    "answer_relevance",
    "correctness",
    "citation_accuracy",
)

# Display names live here so summary.md, the console table and comparisons stay
# in sync. ``{k}`` is replaced with the run's real top_k at render time.
METRIC_LABELS = {
    "precision": "Precision@{k}",
    "recall": "Recall@{k}",
    "video_recall": "Video Recall",
    "hit": "Hit Rate@{k}",
    "mrr": "Mean Reciprocal Rank",
    "ndcg": "Normalized Discounted Cumulative Gain@{k}",
    "judged_precision": "Judged Precision@{k}",
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer Relevance",
    "correctness": "Correctness",
    "citation_accuracy": "Citation Accuracy",
    "abstention_accuracy": "Abstention Accuracy",
    "false_abstention_rate": "False Refusal Rate",
    "retrieval": "Retrieval Latency (ms)",
    "generation": "Generation Latency (ms)",
    "total": "Total Latency (ms)",
    "total_tokens": "Total Tokens",
}

METRIC_DESCRIPTIONS = {
    "precision": "Share of retrieved chunks that are relevant to the question.",
    "recall": "Share of the gold answer moments found among retrieved chunks.",
    "video_recall": "Share of gold videos represented in the retrieved results.",
    "hit": "Share of questions with at least one relevant chunk retrieved.",
    "mrr": "How early the first relevant chunk appears in the ranking.",
    "ndcg": "Ranking quality, rewarding relevant chunks near the top.",
    "judged_precision": (
        "Share of retrieved chunks an LLM judge finds useful, including "
        "non-gold matches."
    ),
    "faithfulness": "Share of answer claims supported by the retrieved context.",
    "answer_relevance": "How directly and completely the answer addresses the question.",
    "correctness": "Share of the reference key points covered by the answer.",
    "citation_accuracy": "Share of citations that correctly support their sentence.",
    "abstention_accuracy": "Share of off-topic questions correctly refused.",
    "false_abstention_rate": "Share of answerable questions wrongly refused.",
    "retrieval": "Mean time spent retrieving context.",
    "generation": "Mean time spent generating the answer.",
    "total": "Mean end-to-end time per question.",
    "total_tokens": "Mean tokens used per question (generator + judge).",
}

SCORE_METRICS = frozenset(RETRIEVAL_METRICS) | frozenset(GENERATION_METRICS) | {
    "abstention_accuracy",
    "false_abstention_rate",
}
PERFORMANCE_METRICS = ("retrieval", "generation", "total", "total_tokens")

# Status bands. Precision is compared with its attainable ceiling first.
GOOD_THRESHOLD = 0.85
OK_THRESHOLD = 0.70
FALSE_REFUSAL_GOOD = 0.05
FALSE_REFUSAL_OK = 0.15
STATUS_TEXT = {
    "good": "✅ Good",
    "ok": "🟡 OK",
    "needs_work": "🔴 Needs work",
    "na": "—",
}
SMALL_SAMPLE_SIZE = 30

# Heuristic targets used only for the plain-English "weakest area" readout.
# Higher is better unless listed in LOWER_IS_BETTER.
TARGETS = {
    "precision": 0.60,
    "recall": 0.80,
    "video_recall": 0.85,
    "hit": 0.85,
    "mrr": 0.70,
    "ndcg": 0.70,
    "judged_precision": 0.70,
    "faithfulness": 0.90,
    "answer_relevance": 0.85,
    "correctness": 0.80,
    "citation_accuracy": 0.90,
    "abstention_accuracy": 0.90,
    "false_abstention_rate": 0.10,
}
LOWER_IS_BETTER = {"false_abstention_rate"}

# P@k is structurally capped when k exceeds the number of gold spans per
# question, so it is reported and explained but never chosen as the single
# "weakest area" for the plain-English readout.
WEAKEST_EXCLUDED = {"precision"}

REMEDIES = {
    "precision": (
        "Retrieved chunks are often not relevant. Lower --top-k, raise --threshold, "
        "or test --rerank; also check that the domain filter matches the question."
    ),
    "recall": (
        "Gold moments are being missed. Lower --threshold, fetch more candidates, "
        "or enable --rerank; for typo/vague questions improve query cleaning first."
    ),
    "video_recall": (
        "The right videos are not surfacing. Check index coverage and the domain "
        "filter, then lower --threshold or add query normalization for hard phrasings."
    ),
    "hit": (
        "Nothing relevant is retrieved for many answerable questions. Lower the "
        "threshold first; if that floods precision, inspect the index and embedding model."
    ),
    "mrr": (
        "Relevant chunks appear, but late. Enable --rerank or disable MMR, and check "
        "that merged/duplicated chunks are not crowding the top of the list."
    ),
    "ndcg": (
        "Ranking quality is weak. Enable --rerank, keep MMR on only if diversity "
        "matters, and inspect cases where gold chunks are just outside the top-k."
    ),
    "judged_precision": (
        "The judge disagrees that non-gold chunks help. Tighten retrieval or use a "
        "stricter gold definition; read the judge reasons in failures.md before tuning."
    ),
    "faithfulness": (
        "Answers contain unsupported or contradicted claims. Keep temperature low, "
        "enforce citations on every claim, and cap the context so the model is not flooded."
    ),
    "answer_relevance": (
        "Answers drift from the question, often on typo/vague inputs. Add glossary or "
        "query rewriting, and inspect the worst questions in failures.md."
    ),
    "correctness": (
        "Reference key points are missing. Improve retrieval for the gold moments, "
        "then tighten the answer prompt to require each key point that the context supports."
    ),
    "citation_accuracy": (
        "Citations do not match the cited chunk. Re-check numbering after context "
        "capping, require one citation per factual sentence, and strip invalid markers."
    ),
    "abstention_accuracy": (
        "The system answers questions it cannot support. Raise the retrieval threshold, "
        "keep the strict not-found instruction, and review off-topic/injection probes."
    ),
    "false_abstention_rate": (
        "The system refuses answerable questions. Lower the threshold, improve typo/vague "
        "query handling, and check that the domain filter is not excluding gold videos."
    ),
}


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def enable_utf8_console() -> None:
    """Keep the status symbols printable on Windows terminals using cp1252."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover - unusual stream
            pass


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "—"
    return f"{number:.{digits}f}"


def fmt_ci(entry: dict[str, Any], digits: int = 3) -> str:
    low, high = entry.get("ci_low"), entry.get("ci_high")
    if low is None or high is None:
        return "—"
    return f"{float(low):.{digits}f} – {float(high):.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    """Render a 0-1 metric as a percentage with one decimal by default."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "—"
    return f"{number * 100:.{digits}f}%"


def fmt_ci_pct(entry: dict[str, Any], digits: int = 1) -> str:
    low, high = entry.get("ci_low"), entry.get("ci_high")
    if low is None or high is None:
        return "—"
    return f"{float(low) * 100:.{digits}f} - {float(high) * 100:.{digits}f}%"


def format_score(metric: str, value: Any) -> str:
    """Display value for a metric: percent, milliseconds, tokens, or raw."""
    if value is None:
        return "—"
    if metric in SCORE_METRICS:
        return fmt_pct(value)
    if metric in ("retrieval", "generation", "total"):
        return f"{float(value):.1f} ms"
    if metric == "total_tokens":
        return f"{int(round(float(value))):,}"
    return fmt(value)


def format_ci(metric: str, entry: dict[str, Any]) -> str:
    if metric in ("retrieval", "generation", "total"):
        low, high = entry.get("ci_low"), entry.get("ci_high")
        if low is None or high is None:
            return "—"
        return f"{float(low):.1f} - {float(high):.1f} ms"
    if metric in SCORE_METRICS:
        return fmt_ci_pct(entry)
    return "—"


def small_sample_warning(n: Any) -> str:
    """A visible caution whenever a score rests on fewer than 30 questions."""
    try:
        count = int(n)
    except (TypeError, ValueError):
        return ""
    if count <= 0 or count >= SMALL_SAMPLE_SIZE:
        return ""
    return f"Small sample (n={count}): treat these scores as a rough indication."


def status_key(
    value: Any,
    metric: str,
    *,
    ceiling: float | None = None,
) -> str:
    """Rate a 0-1 metric as good, ok or needs_work.

    Higher-is-better metrics use 0.85 / 0.70 bands. False Refusal Rate is
    lower-is-better (0.05 / 0.15). Precision is divided by its attainable
    ceiling before applying the higher-is-better bands.
    """
    if value is None:
        return "na"
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "na"
    if metric == "false_abstention_rate":
        if score <= FALSE_REFUSAL_GOOD:
            return "good"
        if score <= FALSE_REFUSAL_OK:
            return "ok"
        return "needs_work"
    ratio = score
    if metric == "precision" and ceiling is not None:
        try:
            ceiling_value = float(ceiling)
        except (TypeError, ValueError):
            ceiling_value = 0.0
        if ceiling_value > 0:
            ratio = score / ceiling_value
    if ratio >= GOOD_THRESHOLD:
        return "good"
    if ratio >= OK_THRESHOLD:
        return "ok"
    return "needs_work"


def status_label(value: Any, metric: str, *, ceiling: float | None = None) -> str:
    return STATUS_TEXT[status_key(value, metric, ceiling=ceiling)]


def _display_width(text: str) -> int:
    """Approximate terminal width so emoji-bearing status cells stay aligned."""
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def _pad(text: str, width: int, *, align: str = "left") -> str:
    padding = " " * max(0, width - _display_width(text))
    return padding + text if align == "right" else text + padding


def console_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    right_align: Iterable[int] = (),
) -> str:
    """Aligned plain-text table for the terminal (no Markdown pipes)."""
    right = set(right_align)
    grid = [[str(value) for value in row] for row in rows]
    widths = [
        max(
            [_display_width(str(headers[index]))]
            + [_display_width(row[index]) for row in grid if index < len(row)]
        )
        for index in range(len(headers))
    ]

    def render(row: Sequence[str]) -> str:
        cells = [
            _pad(value, widths[index], align="right" if index in right else "left")
            for index, value in enumerate(row)
        ]
        return "  ".join(cells).rstrip()

    lines = [render(headers), "  ".join("-" * width for width in widths)]
    lines.extend(render(row) for row in grid)
    return "\n".join(lines)


def cell(value: Any, limit: int = 120) -> str:
    text = str(value if value is not None else "").replace("|", "\\|")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    limit: int = 120,
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value, limit) for value in row) + " |")
    return "\n".join(lines)


def label(metric: str, top_k: int | None = None) -> str:
    """Full display name, with ``{k}`` replaced by the run's real top_k."""
    name = METRIC_LABELS.get(metric, metric)
    if "{k}" not in name:
        return name
    return name.replace("{k}", str(top_k) if top_k else "k")


# --------------------------------------------------------------------------- #
# Extraction and aggregation
# --------------------------------------------------------------------------- #
def retrieval_value(record: dict[str, Any], metric_name: str) -> float | None:
    retrieval = record.get("retrieval") or {}
    value = retrieval.get(metric_name)
    return float(value) if isinstance(value, (int, float)) else None


def generation_value(record: dict[str, Any], metric_name: str) -> float | None:
    entry = (record.get("judge") or {}).get(metric_name) or {}
    value = entry.get("value")
    return float(value) if isinstance(value, (int, float)) else None


def generation_reason(record: dict[str, Any], metric_name: str) -> str:
    entry = (record.get("judge") or {}).get(metric_name) or {}
    if entry.get("error"):
        return f"judge error: {entry['error']}"
    return str(entry.get("reason", "") or "")


def ok_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("status") == "ok"]


def answerable_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in ok_records(records) if record.get("answerable")]


def _metric_block(
    records: Sequence[dict[str, Any]],
    names: Sequence[str],
    extractor: Callable[[dict[str, Any], str], float | None],
) -> dict[str, dict[str, Any]]:
    block: dict[str, dict[str, Any]] = {}
    for name in names:
        values = [extractor(record, name) for record in records]
        if not any(value is not None for value in values):
            continue
        block[name] = metrics.summarize_values(values)
    return block


def abstraction_block(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ok = ok_records(records)
    unanswerable = [record for record in ok if not record.get("answerable", True)]
    answerable = [record for record in ok if record.get("answerable", True)]
    block: dict[str, Any] = {
        "n_unanswerable": len(unanswerable),
        "n_answerable": len(answerable),
        "n_correct_abstentions": sum(
            1 for record in unanswerable if record.get("abstained")
        ),
        "n_false_abstentions": sum(
            1 for record in answerable if record.get("abstained")
        ),
    }
    if unanswerable:
        block["abstention_accuracy"] = metrics.summarize_values(
            [1.0 if record.get("abstained") else 0.0 for record in unanswerable]
        )
    if answerable:
        block["false_abstention_rate"] = metrics.summarize_values(
            [1.0 if record.get("abstained") else 0.0 for record in answerable]
        )
    return block


def latency_block(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    block: dict[str, dict[str, Any]] = {}
    for name in ("retrieval", "generation", "total"):
        values = [
            float((record.get("latency_ms") or {}).get(name, 0) or 0)
            for record in ok_records(records)
            if (record.get("latency_ms") or {}).get(name) is not None
        ]
        if not values:
            continue
        entry = metrics.summarize_values(values, n_resamples=500)
        entry["p50"] = metrics.percentile(values, 50)
        entry["p95"] = metrics.percentile(values, 95)
        entry["min"] = min(values)
        entry["max"] = max(values)
        block[name] = entry
    return block


def usage_block(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    generator = {"input_tokens": 0, "output_tokens": 0}
    judge = {"input_tokens": 0, "output_tokens": 0}
    for record in records:
        for key in ("input_tokens", "output_tokens"):
            generator[key] += int((record.get("usage") or {}).get(key, 0) or 0)
            judge[key] += int((record.get("judge_usage") or {}).get(key, 0) or 0)
    total = {
        key: generator[key] + judge[key] for key in ("input_tokens", "output_tokens")
    }
    for block in (generator, judge, total):
        block["total_tokens"] = block["input_tokens"] + block["output_tokens"]
    return {
        "generator": generator,
        "judge": judge,
        "total": total,
        "n_questions": len(records) or 1,
        "mean_tokens_per_question": round(total["total_tokens"] / (len(records) or 1)),
    }


def infer_top_k(
    records: Sequence[dict[str, Any]], meta: dict[str, Any] | None = None
) -> int | None:
    """The run's top_k from config, falling back to the largest result count."""
    config = (meta or {}).get("config") or {}
    try:
        configured = int(config.get("top_k") or 0)
    except (TypeError, ValueError):
        configured = 0
    if configured > 0:
        return configured
    observed = [
        int((record.get("retrieval") or {}).get("n_retrieved") or 0)
        for record in answerable_records(records)
    ]
    return max(observed) if observed and max(observed) > 0 else None


def precision_ceiling(
    records: Sequence[dict[str, Any]], top_k: int | None
) -> dict[str, Any]:
    """Mean attainable Precision@k when top_k exceeds the gold chunk count.

    Each question is capped at ``min(1, gold chunks / top_k)``; the mean of
    those caps is the ceiling a perfect retriever could reach on this dataset.
    """
    block: dict[str, Any] = {
        "value": None,
        "n": 0,
        "top_k": top_k,
        "formula": "min(1, gold chunks / top_k)",
    }
    if not top_k or top_k <= 0:
        return block
    values: list[float] = []
    for record in answerable_records(records):
        retrieval = record.get("retrieval") or {}
        n_gold = retrieval.get("n_gold")
        if n_gold is None:
            n_gold = len(record.get("gold_sources") or [])
        try:
            gold_count = float(n_gold or 0)
        except (TypeError, ValueError):
            gold_count = 0.0
        if gold_count <= 0:
            continue
        values.append(min(1.0, gold_count / float(top_k)))
    if values:
        block["value"] = sum(values) / len(values)
        block["n"] = len(values)
    return block


def summary_top_k(summary: dict[str, Any]) -> int | None:
    """Best available top_k for display names and precision status."""
    config = (summary.get("meta") or {}).get("config") or {}
    try:
        configured = int(config.get("top_k") or 0)
    except (TypeError, ValueError):
        configured = 0
    if configured > 0:
        return configured
    ceiling = (summary.get("overall") or {}).get("precision_ceiling") or {}
    try:
        from_ceiling = int(ceiling.get("top_k") or 0)
    except (TypeError, ValueError):
        from_ceiling = 0
    return from_ceiling or None


def group_block(
    records: Sequence[dict[str, Any]], *, top_k: int | None = None
) -> dict[str, Any]:
    answerable = answerable_records(records)
    return {
        "n_questions": len(records),
        "n_errors": sum(1 for record in records if record.get("status") != "ok"),
        "retrieval": _metric_block(answerable, RETRIEVAL_METRICS, retrieval_value),
        "generation": _metric_block(answerable, GENERATION_METRICS, generation_value),
        "abstention": abstraction_block(records),
        "precision_ceiling": precision_ceiling(answerable, top_k),
        "latency": latency_block(records),
        "usage": usage_block(records),
    }


def build_summary(
    records: Sequence[dict[str, Any]],
    meta: dict[str, Any] | None = None,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate records into overall, per-domain and per-category numbers."""
    meta = meta or {}
    answerable = answerable_records(records)
    ok = ok_records(records)
    top_k = infer_top_k(records, meta)

    overall = {
        "retrieval": _metric_block(answerable, RETRIEVAL_METRICS, retrieval_value),
        "generation": _metric_block(answerable, GENERATION_METRICS, generation_value),
        "abstention": abstraction_block(records),
        "precision_ceiling": precision_ceiling(answerable, top_k),
        "latency": latency_block(records),
        "usage": usage_block(records),
    }

    by_domain: dict[str, Any] = {}
    for domain in ("dsa", "system_design"):
        subset = [
            record
            for record in records
            if record.get("domain") == domain and record.get("category") != "off_topic"
        ]
        if subset:
            by_domain[domain] = group_block(subset, top_k=top_k)

    by_category: dict[str, Any] = {}
    categories = sorted({str(record.get("category", "")) for record in records})
    for category in categories:
        if not category:
            continue
        subset = [record for record in records if record.get("category") == category]
        if subset:
            by_category[category] = group_block(subset, top_k=top_k)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "n_questions": len(records),
        "n_ok": len(ok),
        "n_errors": len(records) - len(ok),
        "n_answerable": len(answerable),
        "n_unanswerable": sum(1 for record in ok if not record.get("answerable")),
        "overall": overall,
        "by_domain": by_domain,
        "by_category": by_category,
        "weakest": None,
        "errors": [
            {"id": record.get("id"), "error": record.get("error")}
            for record in records
            if record.get("status") != "ok"
        ],
    }
    summary["weakest"] = weakest_metric(summary)
    summary["readout"] = plain_english_readout(summary)
    return summary


# --------------------------------------------------------------------------- #
# Weakest metric / readout
# --------------------------------------------------------------------------- #
def _overall_values(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    overall = summary.get("overall") or {}
    values: dict[str, dict[str, Any]] = {}
    for block_name in ("retrieval", "generation", "abstention"):
        for name, entry in (overall.get(block_name) or {}).items():
            if isinstance(entry, dict) and entry.get("value") is not None:
                values[name] = entry
    return values


def weakest_metric(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Metric furthest below its heuristic target, normalized by target size."""
    worst: dict[str, Any] | None = None
    top_k = summary_top_k(summary)
    for name, entry in _overall_values(summary).items():
        target = TARGETS.get(name)
        if name in WEAKEST_EXCLUDED or target is None or not entry.get("n"):
            continue
        value = float(entry["value"])
        if name in LOWER_IS_BETTER:
            severity = max(0.0, (value - target) / max(target, 1e-9))
        else:
            severity = max(0.0, (target - value) / max(target, 1e-9))
        candidate = {
            "metric": name,
            "label": label(name, top_k),
            "value": value,
            "target": target,
            "direction": "lower_is_better" if name in LOWER_IS_BETTER else "higher_is_better",
            "severity": round(severity, 4),
            "n": entry.get("n"),
        }
        if worst is None or severity > worst["severity"]:
            worst = candidate
    if worst is not None and worst["severity"] <= 0:
        return None
    return worst


def _precision_note(summary: dict[str, Any]) -> str:
    """Explain a capped Precision@k instead of mislabeling it as the weakest area."""
    entry = ((summary.get("overall") or {}).get("retrieval") or {}).get("precision") or {}
    if entry.get("value") is None or not entry.get("n"):
        return ""
    value = float(entry["value"])
    target = TARGETS["precision"]
    if value >= target:
        return ""
    top_k = summary_top_k(summary)
    return (
        f" {label('precision', top_k)} is also below its nominal target "
        f"({fmt_pct(value)} vs {fmt_pct(target)}), but it is structurally capped "
        "when top_k is larger than the number of gold chunks per question. The "
        "report rates it against that attainable ceiling instead."
    )


def plain_english_readout(summary: dict[str, Any]) -> str:
    weakest = summary.get("weakest") or weakest_metric(summary)
    note = _precision_note(summary)
    if not weakest:
        base = "No ranked metric is below its heuristic target."
        tail = "Read failures.md to check the tail cases before trusting this result."
        return f"{base}{note} {tail}" if note else f"{base} {tail}"
    remedy = REMEDIES.get(
        str(weakest["metric"]),
        "Inspect the worst questions in failures.md and tune the pipeline setting "
        "most directly related to this metric.",
    )
    return (
        f"Weakest area: {weakest['label']} = {fmt_pct(weakest['value'])} "
        f"(target {fmt_pct(weakest['target'])}; n={weakest['n']}). {remedy}"
        + note
    )


# --------------------------------------------------------------------------- #
# Display rows shared by summary.md and the console
# --------------------------------------------------------------------------- #
OVERALL_HEADERS = ["Metric", "Score", "95% CI", "n", "What it measures", "Status"]


def _entry(block: dict[str, Any] | None, name: str) -> dict[str, Any]:
    entry = (block or {}).get(name)
    return entry if isinstance(entry, dict) else {}


def _metric_row(
    name: str,
    entry: dict[str, Any],
    top_k: int | None,
    *,
    ceiling: float | None = None,
    performance: bool = False,
) -> list[Any]:
    if performance:
        score = format_score(name, entry.get("value"))
        ci = format_ci(name, entry)
        status = "—"
    else:
        score = fmt_pct(entry.get("value"))
        ci = fmt_ci_pct(entry)
        status = status_label(
            entry.get("value"),
            name,
            ceiling=ceiling if name == "precision" else None,
        )
    return [
        label(name, top_k),
        score,
        ci,
        int(entry.get("n") or 0),
        METRIC_DESCRIPTIONS.get(name, ""),
        status,
    ]


def _overall_rows(
    summary: dict[str, Any], block_name: str
) -> list[list[Any]]:
    overall = summary.get("overall") or {}
    top_k = summary_top_k(summary)
    ceiling = ((overall.get("precision_ceiling") or {}).get("value"))
    if block_name == "retrieval":
        block = overall.get("retrieval") or {}
        return [
            _metric_row(name, _entry(block, name), top_k, ceiling=ceiling)
            for name in RETRIEVAL_METRICS
        ]
    if block_name == "generation":
        block = overall.get("generation") or {}
        return [
            _metric_row(name, _entry(block, name), top_k)
            for name in GENERATION_METRICS
        ]
    if block_name == "abstention":
        block = overall.get("abstention") or {}
        return [
            _metric_row(name, _entry(block, name), top_k)
            for name in ("abstention_accuracy", "false_abstention_rate")
        ]
    usage = overall.get("usage") or {}
    rows = [
        _metric_row(
            name,
            _entry(overall.get("latency") or {}, name),
            top_k,
            performance=True,
        )
        for name in ("retrieval", "generation", "total")
    ]
    rows.append(
        _metric_row(
            "total_tokens",
            {
                "value": usage.get("mean_tokens_per_question"),
                "n": summary.get("n_questions", 0),
            },
            top_k,
            performance=True,
        )
    )
    return rows


def _smallest_n(rows: Sequence[Sequence[Any]]) -> int:
    values = []
    for row in rows:
        try:
            value = int(row[3])
        except (IndexError, TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    return min(values) if values else 0


def _precision_ceiling_note(summary: dict[str, Any]) -> str:
    overall = summary.get("overall") or {}
    ceiling = overall.get("precision_ceiling") or {}
    top_k = summary_top_k(summary)
    name = label("precision", top_k)
    value = ceiling.get("value")
    if value is not None:
        return (
            f"*Precision ceiling:* {name} cannot exceed the number of gold chunks "
            f"divided by top_k. On this run the mean attainable ceiling is "
            f"{fmt_pct(value)} (mean of min(1, gold chunks / {top_k or 'k'}) across "
            f"answerable questions, n={ceiling.get('n', 0)}). The Status column "
            f"compares {name} with that ceiling, not with 100%."
        )
    return (
        f"*Precision ceiling:* {name} is capped at min(1, gold chunks / top_k). "
        "No ceiling could be computed for this run, so the status uses the raw score."
    )


def _abstention_notes(summary: dict[str, Any]) -> list[str]:
    abstention = (summary.get("overall") or {}).get("abstention") or {}
    notes = [
        f"Correct abstentions: {abstention.get('n_correct_abstentions', 0)} / "
        f"{abstention.get('n_unanswerable', 0)} unanswerable. "
        f"False refusals: {abstention.get('n_false_abstentions', 0)} / "
        f"{abstention.get('n_answerable', 0)} answerable."
    ]
    if not abstention.get("n_unanswerable"):
        notes.append(
            "_No unanswerable or off-topic questions in this run, so Abstention "
            "Accuracy is not available._"
        )
    if not abstention.get("n_answerable"):
        notes.append(
            "_No answerable questions in this run, so False Refusal Rate is not "
            "available._"
        )
    return notes


def _generation_notes(summary: dict[str, Any]) -> list[str]:
    block = (summary.get("overall") or {}).get("generation") or {}
    if any(_entry(block, name).get("value") is not None for name in GENERATION_METRICS):
        return []
    return [
        "_Generation metrics were not measured in this run (judge disabled or "
        "every answer abstained)._"
    ]


def _retrieval_notes(summary: dict[str, Any]) -> list[str]:
    notes = [_precision_ceiling_note(summary)]
    block = (summary.get("overall") or {}).get("retrieval") or {}
    if _entry(block, "judged_precision").get("value") is None:
        notes.append(
            "_Judged Precision is only measured with `--judge-retrieval`._"
        )
    return notes


def _section_notes(summary: dict[str, Any], block_name: str) -> list[str]:
    if block_name == "retrieval":
        return _retrieval_notes(summary)
    if block_name == "generation":
        return _generation_notes(summary)
    if block_name == "abstention":
        return _abstention_notes(summary)
    return ["_Performance rows are descriptive and have no pass/fail status._"]


def _status_counts(summary: dict[str, Any]) -> dict[str, int]:
    counts = {"good": 0, "ok": 0, "needs_work": 0}
    ceiling = ((summary.get("overall") or {}).get("precision_ceiling") or {}).get("value")
    for name, entry in _overall_values(summary).items():
        if name not in SCORE_METRICS:
            continue
        key = status_key(
            entry.get("value"),
            name,
            ceiling=ceiling if name == "precision" else None,
        )
        if key in counts:
            counts[key] += 1
    return counts


def headline(summary: dict[str, Any]) -> str:
    config = (summary.get("meta") or {}).get("config") or {}
    top_k = summary_top_k(summary)
    counts = _status_counts(summary)
    paragraphs = [
        f"{summary.get('n_questions', 0)} questions "
        f"({summary.get('n_answerable', 0)} answerable, "
        f"{summary.get('n_unanswerable', 0)} unanswerable) on the "
        f"{config.get('split') or 'all'} split with top_k={top_k or '—'}; "
        f"{summary.get('n_errors', 0)} errors."
    ]
    if any(counts.values()):
        paragraphs.append(
            f"Status: {STATUS_TEXT['good']} {counts['good']} · "
            f"{STATUS_TEXT['ok']} {counts['ok']} · "
            f"{STATUS_TEXT['needs_work']} {counts['needs_work']}."
        )
    else:
        paragraphs.append("Status: no scored metrics in this run.")
    readout = summary.get("readout")
    if readout:
        paragraphs.append(str(readout))
    return "\n\n".join(paragraphs)


def _on_off(value: Any) -> str:
    if value is None:
        return "—"
    return "on" if bool(value) else "off"


def _run_info_rows(summary: dict[str, Any]) -> list[list[Any]]:
    meta = summary.get("meta") or {}
    config = meta.get("config") or {}
    generator = meta.get("generator") or {}
    judge = meta.get("judge") or {}
    snapshot = meta.get("index_snapshot") or {}
    if snapshot:
        index_text = (
            f"{snapshot.get('dsa_videos', 0)} DSA videos "
            f"({snapshot.get('dsa_chunks', 0)} chunks), "
            f"{snapshot.get('system_design_videos', 0)} System Design videos "
            f"({snapshot.get('system_design_chunks', 0)} chunks)"
        )
    else:
        index_text = "—"
    threshold = config.get("threshold")
    min_overlap = config.get("min_overlap")
    return [
        ["Run name", config.get("config_name") or "evaluation"],
        ["Generated at", summary.get("generated_at") or "—"],
        ["Run started", meta.get("created_at") or "—"],
        ["Split", config.get("split") or "all"],
        [
            "Questions",
            f"{summary.get('n_questions', 0)} "
            f"({summary.get('n_answerable', 0)} answerable, "
            f"{summary.get('n_unanswerable', 0)} unanswerable)",
        ],
        ["Errors", summary.get("n_errors", 0)],
        ["Domain filter", config.get("domain") or "all"],
        ["top_k", config.get("top_k") or "—"],
        ["Score threshold", fmt(threshold, 2) if threshold is not None else "—"],
        ["MMR", _on_off(config.get("use_mmr"))],
        ["Reranker", _on_off(config.get("use_reranker"))],
        [
            "Minimum overlap",
            f"{fmt(min_overlap, 1)} s" if min_overlap is not None else "—",
        ],
        ["Transport", "API" if config.get("via_api") else "in-process"],
        [
            "Generator",
            f"{generator.get('provider', '?')}/{generator.get('model', '?')}",
        ],
        [
            "Judge",
            f"{judge.get('provider', '—')}/{judge.get('model', '—')}"
            if judge
            else "disabled",
        ],
        ["Config hash", meta.get("config_hash") or "—"],
        ["Index", index_text],
    ]


def _group_entry(group: dict[str, Any], name: str) -> dict[str, Any]:
    if name in RETRIEVAL_METRICS:
        return _entry(group.get("retrieval") or {}, name)
    if name in GENERATION_METRICS:
        return _entry(group.get("generation") or {}, name)
    return _entry(group.get("abstention") or {}, name)


def _matrix(
    groups: dict[str, Any],
    group_label: Callable[[str], str],
    top_k: int | None,
) -> str:
    if not groups:
        return "_not measured in this run_"
    names = [
        name
        for name in (
            list(RETRIEVAL_METRICS)
            + list(GENERATION_METRICS)
            + ["abstention_accuracy", "false_abstention_rate"]
        )
        if any(
            _group_entry(group, name).get("value") is not None
            for group in groups.values()
        )
    ]
    headers = ["Metric"] + [
        f"{group_label(key)} (n={group.get('n_questions', 0)})"
        for key, group in groups.items()
    ]
    rows = [
        [label(name, top_k)]
        + [fmt_pct(_group_entry(group, name).get("value")) for group in groups.values()]
        for name in names
    ]
    text = table(headers, rows)
    smallest = min(
        (int(group.get("n_questions") or 0) for group in groups.values()),
        default=0,
    )
    warning = small_sample_warning(smallest)
    return f"{text}\n\n{warning}" if warning else text


def _domain_label(domain: str) -> str:
    return {"dsa": "DSA", "system_design": "System Design"}.get(
        domain, domain.replace("_", " ").title()
    )


def _category_label(category: str) -> str:
    return category.replace("_", " ").title()


def _latency_detail_table(summary: dict[str, Any]) -> str:
    latency = (summary.get("overall") or {}).get("latency") or {}
    if not latency:
        return "_no latency data_"
    top_k = summary_top_k(summary)
    return table(
        ["Statistic", "Mean", "95% CI", "p50", "p95", "min", "max", "n"],
        [
            [
                label(name, top_k),
                fmt(entry.get("value"), 1),
                fmt_ci(entry, 1),
                fmt(entry.get("p50"), 1),
                fmt(entry.get("p95"), 1),
                fmt(entry.get("min"), 1),
                fmt(entry.get("max"), 1),
                entry.get("n"),
            ]
            for name, entry in latency.items()
        ],
    )


def _usage_table(summary: dict[str, Any]) -> str:
    usage = (summary.get("overall") or {}).get("usage") or {}
    rows = [
        [
            name.title(),
            block.get("input_tokens", 0),
            block.get("output_tokens", 0),
            block.get("total_tokens", 0),
        ]
        for name, block in usage.items()
        if isinstance(block, dict) and "input_tokens" in block
    ]
    return table(["Usage", "Input", "Output", "Total"], rows) if rows else "_no usage data_"


def render_markdown(summary: dict[str, Any]) -> str:
    config = (summary.get("meta") or {}).get("config") or {}
    name = config.get("config_name") or "evaluation"
    top_k = summary_top_k(summary)

    lines = [
        f"# RAG evaluation — {name}",
        "",
        f"Generated {summary.get('generated_at', '')}",
        "",
        "## Headline",
        "",
    ]
    for paragraph in headline(summary).split("\n\n"):
        lines.extend([paragraph, ""])
    lines.extend(
        [
            "## Run info",
            "",
            table(["Setting", "Value"], _run_info_rows(summary)),
            "",
            "## Overall",
            "",
        ]
    )
    for block_name, title in (
        ("retrieval", "Retrieval (answerable questions only)"),
        ("generation", "Generation"),
        ("abstention", "Abstention"),
        ("performance", "Performance"),
    ):
        rows = _overall_rows(summary, block_name)
        lines.extend([f"### {title}", "", table(OVERALL_HEADERS, rows), ""])
        for note in _section_notes(summary, block_name):
            if note:
                lines.extend([note, ""])
        warning = small_sample_warning(_smallest_n(rows))
        if warning:
            lines.extend([warning, ""])
        if block_name == "performance":
            lines.extend(
                [
                    "#### Latency detail",
                    "",
                    _latency_detail_table(summary),
                    "",
                    "#### Token usage",
                    "",
                    _usage_table(summary),
                    "",
                ]
            )
    lines.extend(
        [
            "## By domain",
            "",
            _matrix(summary.get("by_domain") or {}, _domain_label, top_k),
            "",
            "## By category",
            "",
            _matrix(summary.get("by_category") or {}, _category_label, top_k),
            "",
            "## Glossary",
            "",
            table(
                ["Term", "Meaning"],
                [
                    [
                        "Precision",
                        "Of the chunks retrieved, the share that overlap a gold "
                        "answer moment. It is capped when top_k exceeds the number "
                        "of gold chunks.",
                    ],
                    [
                        "Recall",
                        "Of the gold answer moments, the share that appear among "
                        "the retrieved chunks.",
                    ],
                    [
                        "Faithfulness",
                        "Of the claims in an answer, the share supported by the "
                        "retrieved context rather than invented.",
                    ],
                    [
                        "Relevance",
                        "How directly the answer addresses the question asked, "
                        "including whether it stays on topic.",
                    ],
                    [
                        "Correctness",
                        "How many of the reference answer's key points the answer "
                        "covers, with a penalty for contradictions.",
                    ],
                ],
                limit=400,
            ),
            "",
            "## Readout",
            "",
            summary.get("readout", ""),
            "",
            f"Detailed worst cases: `failures.md`. Errors: {summary.get('n_errors', 0)}.",
            "",
        ]
    )
    return "\n".join(lines)


def render_console(summary: dict[str, Any]) -> str:
    """Print the same grouped metrics table in aligned plain text."""
    config = (summary.get("meta") or {}).get("config") or {}
    name = config.get("config_name") or "evaluation"
    lines = [f"RAG evaluation — {name}", ""]
    for paragraph in headline(summary).split("\n\n"):
        lines.extend([paragraph, ""])
    for block_name, title in (
        ("retrieval", "Retrieval"),
        ("generation", "Generation"),
        ("abstention", "Abstention"),
        ("performance", "Performance"),
    ):
        rows = _overall_rows(summary, block_name)
        lines.extend(
            [
                title,
                "",
                console_table(OVERALL_HEADERS, rows, right_align={1, 2, 3}),
                "",
            ]
        )
        for note in _section_notes(summary, block_name):
            if note:
                lines.extend([note, ""])
        warning = small_sample_warning(_smallest_n(rows))
        if warning:
            lines.extend([warning, ""])
    lines.extend(["Readout", "", summary.get("readout", ""), ""])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# failures.md
# --------------------------------------------------------------------------- #
def _worst_retrieval(
    records: Sequence[dict[str, Any]], metric_name: str, limit: int = 10
) -> list[tuple[float, dict[str, Any]]]:
    items = []
    for record in answerable_records(records):
        value = retrieval_value(record, metric_name)
        if value is not None:
            items.append((value, record))
    items.sort(key=lambda pair: (pair[0], str(pair[1].get("id"))))
    return items[:limit]


def _worst_generation(
    records: Sequence[dict[str, Any]], metric_name: str, limit: int = 10
) -> list[tuple[float, dict[str, Any]]]:
    items = []
    for record in answerable_records(records):
        value = generation_value(record, metric_name)
        if value is not None:
            items.append((value, record))
    items.sort(key=lambda pair: (pair[0], str(pair[1].get("id"))))
    return items[:limit]


def _retrieval_why(record: dict[str, Any]) -> str:
    gold = record.get("gold_sources") or []
    retrieved = record.get("retrieved") or []
    gold_text = ", ".join(
        f"{source.get('video_id')}@{source.get('start')}-{source.get('end')}s"
        for source in gold
    )
    retrieved_text = "; ".join(
        f"{item.get('video_id')}@{item.get('start')}-{item.get('end')}s"
        for item in retrieved[:5]
    )
    return f"gold: {gold_text} | got: {retrieved_text or 'none'}"


def render_failures(records: Sequence[dict[str, Any]], summary: dict[str, Any]) -> str:
    meta = summary.get("meta") or {}
    name = (meta.get("config") or {}).get("config_name") or "evaluation"
    lines = [f"# Failure digest — {name}", ""]

    errors = [record for record in records if record.get("status") != "ok"]
    lines.extend([f"## Run errors ({len(errors)})", ""])
    if errors:
        lines.append(
            table(
                ["id", "question", "error"],
                [
                    [
                        record.get("id"),
                        record.get("question"),
                        (record.get("error") or "")[:300],
                    ]
                    for record in errors
                ],
            )
        )
    else:
        lines.append("_none_")
    lines.append("")

    false_abstentions = [
        record
        for record in answerable_records(records)
        if record.get("abstained")
    ]
    missed_abstentions = [
        record
        for record in ok_records(records)
        if not record.get("answerable") and not record.get("abstained")
    ]
    lines.extend([f"## False abstentions ({len(false_abstentions)})", ""])
    if false_abstentions:
        lines.append(
            table(
                ["id", "category", "question", "answer"],
                [
                    [
                        record.get("id"),
                        record.get("category"),
                        record.get("question"),
                        (record.get("answer") or "")[:200],
                    ]
                    for record in false_abstentions
                ],
            )
        )
    else:
        lines.append("_none_")
    lines.extend(["", f"## Missed abstentions ({len(missed_abstentions)})", ""])
    if missed_abstentions:
        lines.append(
            table(
                ["id", "question", "answer"],
                [
                    [
                        record.get("id"),
                        record.get("question"),
                        (record.get("answer") or "")[:240],
                    ]
                    for record in missed_abstentions
                ],
            )
        )
    else:
        lines.append("_none_")
    lines.append("")

    top_k = summary_top_k(summary)
    lines.extend(["## Retrieval", ""])
    for metric_name in RETRIEVAL_METRICS:
        worst = _worst_retrieval(records, metric_name)
        if not worst:
            continue
        lines.extend([f"### {label(metric_name, top_k)} — 10 worst", ""])
        lines.append(
            table(
                ["id", "category", "value", "question", "why"],
                [
                    [
                        record.get("id"),
                        record.get("category"),
                        fmt(value),
                        record.get("question"),
                        _retrieval_why(record),
                    ]
                    for value, record in worst
                ],
            )
        )
        lines.append("")

    lines.extend(["## Generation", ""])
    judge_errors = []
    for metric_name in GENERATION_METRICS:
        worst = _worst_generation(records, metric_name)
        for _value, record in worst:
            entry = (record.get("judge") or {}).get(metric_name) or {}
            if entry.get("error"):
                judge_errors.append((record.get("id"), metric_name, entry["error"]))
        if not worst:
            continue
        lines.extend([f"### {label(metric_name, top_k)} — 10 worst", ""])
        lines.append(
            table(
                ["id", "category", "value", "question", "judge reason"],
                [
                    [
                        record.get("id"),
                        record.get("category"),
                        fmt(value),
                        record.get("question"),
                        generation_reason(record, metric_name) or "—",
                    ]
                    for value, record in worst
                ],
            )
        )
        lines.append("")

    if judge_errors:
        lines.extend([f"## Judge errors ({len(judge_errors)})", ""])
        lines.append(
            table(
                ["id", "metric", "error"],
                [[qid, metric_name, error[:300]] for qid, metric_name, error in judge_errors],
            )
        )
        lines.append("")

    slowest = sorted(
        ok_records(records),
        key=lambda record: float(
            (record.get("latency_ms") or {}).get("total", 0) or 0
        ),
        reverse=True,
    )[:10]
    lines.extend(["## Slowest questions", ""])
    if slowest:
        lines.append(
            table(
                ["id", "total ms", "retrieval ms", "generation ms", "question"],
                [
                    [
                        record.get("id"),
                        fmt((record.get("latency_ms") or {}).get("total"), 0),
                        fmt((record.get("latency_ms") or {}).get("retrieval"), 0),
                        fmt((record.get("latency_ms") or {}).get("generation"), 0),
                        record.get("question"),
                    ]
                    for record in slowest
                ],
            )
        )
    else:
        lines.append("_none_")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Write / read / compare
# --------------------------------------------------------------------------- #
def write_reports(
    records: Sequence[dict[str, Any]],
    meta: dict[str, Any] | None,
    out_dir: Path,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Write summary.json, summary.md and failures.md; return the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(records, meta, n_resamples=n_resamples, seed=seed)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    (out_dir / "failures.md").write_text(
        render_failures(records, summary), encoding="utf-8"
    )
    return summary


def read_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
    return rows


def load_run(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = run_dir.resolve()
    records_path = run_dir / "questions.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing {records_path}")
    records = read_records(records_path)
    meta: dict[str, Any] = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        meta = {
            key: payload.get(key)
            for key in (
                "config",
                "config_hash",
                "index_snapshot",
                "generator",
                "judge",
                "estimate",
                "created_at",
            )
        }
    return records, meta


def load_summary(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_dir():
        path = path / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_overall(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    flat: dict[str, dict[str, Any]] = {}
    overall = summary.get("overall") or {}
    for block_name in ("retrieval", "generation", "abstention"):
        for name, entry in (overall.get(block_name) or {}).items():
            if isinstance(entry, dict) and entry.get("value") is not None:
                flat[name] = entry
    for name, entry in (overall.get("latency") or {}).items():
        if isinstance(entry, dict) and entry.get("value") is not None:
            flat[name] = entry
    usage = (overall.get("usage") or {}).get("total") or {}
    if usage:
        flat["total_tokens"] = {"value": usage.get("total_tokens", 0), "n": 1}
    return flat


def _format_delta(name: str, delta: float | None) -> str:
    if delta is None:
        return "—"
    if name in SCORE_METRICS:
        return f"{delta * 100:+.1f} pp"
    if name in ("retrieval", "generation", "total"):
        return f"{delta:+.1f} ms"
    if name == "total_tokens":
        return f"{delta:+,.0f} tokens"
    return f"{delta:+.3f}"


def compare_summaries(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Markdown table of metric deltas (new - old) with a verdict column."""
    old_values = _flatten_overall(old)
    new_values = _flatten_overall(new)
    names = list(dict.fromkeys(list(old_values) + list(new_values)))
    top_k = summary_top_k(new) or summary_top_k(old)
    rows = []
    for name in names:
        old_entry = old_values.get(name) or {}
        new_entry = new_values.get(name) or {}
        old_value = old_entry.get("value")
        new_value = new_entry.get("value")
        if old_value is None and new_value is None:
            continue
        if old_value is None or new_value is None:
            delta = None
            verdict = "only one side measured"
        else:
            delta = float(new_value) - float(old_value)
            lower_better = (
                name in LOWER_IS_BETTER
                or name in ("retrieval", "generation", "total", "total_tokens")
            )
            if abs(delta) < 1e-12:
                verdict = "same"
            elif (delta < 0) == lower_better:
                verdict = "better"
            else:
                verdict = "worse"
        rows.append(
            [
                label(name, top_k),
                format_score(name, old_value),
                format_score(name, new_value),
                _format_delta(name, delta),
                verdict,
            ]
        )
    old_name = (old.get("meta") or {}).get("config", {}).get("config_name") or "old"
    new_name = (new.get("meta") or {}).get("config", {}).get("config_name") or "new"
    return "\n".join(
        [
            f"# Comparison: {old_name} → {new_name}",
            "",
            table(["Metric", old_name, new_name, "Delta", "Verdict"], rows),
            "",
            f"Questions: {old.get('n_questions', 0)} → {new.get('n_questions', 0)}; "
            f"errors: {old.get('n_errors', 0)} → {new.get('n_errors', 0)}.",
            "",
        ]
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render or compare YouTube Reviser evaluation reports."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="results directory to (re)render",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("OLD", "NEW"),
        help="compare two results directories (or summary.json files)",
    )
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    enable_utf8_console()
    args = parse_args(argv)
    try:
        if args.compare:
            old = load_summary(args.compare[0])
            new = load_summary(args.compare[1])
            print(compare_summaries(old, new))
            return
        if not args.run_dir:
            raise ValueError("Pass a results directory or --compare OLD NEW.")
        records, meta = load_run(args.run_dir)
        summary = write_reports(
            records,
            meta,
            args.run_dir,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
        print(f"Wrote {args.run_dir / 'summary.md'}")
        print(f"Wrote {args.run_dir / 'summary.json'}")
        print(f"Wrote {args.run_dir / 'failures.md'}")
        print()
        print(render_console(summary))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
