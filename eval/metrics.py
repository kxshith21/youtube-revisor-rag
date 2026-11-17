"""Pure metric functions for the YouTube Reviser evaluation harness.

No network, no LLM, no file I/O. Everything here operates on small dataclasses
or plain dicts so it can be unit-tested with hand-built examples.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

# --------------------------------------------------------------------------- #
# Spans and relevance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Span:
    """A video time range. Used for both retrieved and gold chunks."""

    video_id: str
    start: float
    end: float


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the intersection of two time ranges, or 0 when disjoint."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def is_relevant(retrieved: Span, gold: Span, min_overlap: float = 10.0) -> bool:
    """A retrieved chunk is relevant when it is from a gold video and overlaps
    the gold time range by at least ``min_overlap`` seconds."""
    if retrieved.video_id != gold.video_id:
        return False
    return overlap_seconds(retrieved.start, retrieved.end, gold.start, gold.end) >= min_overlap


def span_from_dict(value: dict[str, Any]) -> Span:
    """Build a Span from a chunk-like dict (gold source or retrieved result)."""
    return Span(
        video_id=str(value.get("video_id", "")),
        start=float(value.get("start", 0) or 0),
        end=float(value.get("end", 0) or 0),
    )


def matched_gold_indices(
    retrieved: Sequence[Span], gold: Sequence[Span], min_overlap: float = 10.0
) -> list[int]:
    """Gold indices covered by each retrieved chunk (empty list = unmatched)."""
    return [
        i
        for i, gold_span in enumerate(gold)
        if any(is_relevant(r, gold_span, min_overlap) for r in retrieved)
    ]


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def retrieval_metrics(
    retrieved: Sequence[Span],
    gold: Sequence[Span],
    k: int,
    min_overlap: float = 10.0,
) -> dict[str, float | int | None]:
    """Precision, recall, video recall, hit, MRR and nDCG for one question.

    ``k`` is the requested top-k (used to cap nDCG's ideal ranking). The
    actual number returned is used as the precision denominator, per spec.
    """
    n_retrieved = len(retrieved)
    n_gold = len(gold)
    if n_gold == 0:
        return {
            "n_gold": 0,
            "n_retrieved": n_retrieved,
            "n_relevant_retrieved": 0,
            "precision": None,
            "recall": None,
            "video_recall": None,
            "hit": None,
            "mrr": None,
            "ndcg": None,
        }

    relevance = [
        any(is_relevant(r, g, min_overlap) for g in gold) for r in retrieved
    ]
    relevant_count = sum(1 for flag in relevance if flag)
    covered_gold = sum(
        1 for g in gold if any(is_relevant(r, g, min_overlap) for r in retrieved)
    )
    gold_videos = {g.video_id for g in gold}
    covered_videos = {r.video_id for r in retrieved if r.video_id in gold_videos}

    first_rank = next((i for i, flag in enumerate(relevance, start=1) if flag), None)
    mrr = 1.0 / first_rank if first_rank else 0.0

    # Binary nDCG: each retrieved relevant chunk has gain 1; the ideal ranking
    # has one gain per gold source, capped at k.
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, flag in enumerate(relevance[: max(1, k)], start=1)
        if flag
    )
    ideal_n = min(max(1, k), n_gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "n_gold": n_gold,
        "n_retrieved": n_retrieved,
        "n_relevant_retrieved": relevant_count,
        # 0/0 is undefined; treat "no results for an answerable question" as 0.
        "precision": relevant_count / n_retrieved if n_retrieved else 0.0,
        "recall": covered_gold / n_gold,
        "video_recall": len(covered_videos) / len(gold_videos),
        "hit": 1.0 if relevant_count else 0.0,
        "mrr": mrr,
        "ndcg": ndcg,
    }


def judged_precision(
    matched: Sequence[bool],
    judged_relevant: Sequence[bool | None],
) -> float | None:
    """Precision after an LLM judges chunks that gold did not match.

    ``matched`` and ``judged_relevant`` are aligned with the retrieved list.
    Matched chunks count as relevant; unmatched chunks use the judge verdict.
    """
    if len(matched) != len(judged_relevant):
        raise ValueError("matched and judged_relevant must have the same length")
    if not matched:
        return 0.0
    relevant = sum(
        1 for is_match, judged in zip(matched, judged_relevant) if is_match or judged
    )
    return relevant / len(matched)


# --------------------------------------------------------------------------- #
# Abstention
# --------------------------------------------------------------------------- #
ABSTAIN_MARKERS = (
    "couldn't find this in the indexed videos",
    "could not find this in the indexed videos",
    "couldn't find this",
    "could not find this",
    "not in the indexed videos",
    "no relevant content found",
    "i couldn't find",
    "i could not find",
)


def is_abstention(answer: str, found: bool) -> bool:
    """True when the pipeline refused to answer.

    ``found=False`` is the pipeline's own signal; the text markers catch cases
    where a model phrases a refusal differently.
    """
    if not found:
        return True
    text = (answer or "").casefold()
    return any(marker in text for marker in ABSTAIN_MARKERS)


def abstention_metrics(records: Iterable[dict[str, Any]]) -> dict[str, float | int | None]:
    """Abstention accuracy on unanswerable questions and false abstention on
    answerable questions."""
    unanswerable = [r for r in records if not r.get("answerable", True)]
    answerable = [r for r in records if r.get("answerable", True)]
    correct = sum(1 for r in unanswerable if r.get("abstained", False))
    false_abstentions = sum(1 for r in answerable if r.get("abstained", False))
    return {
        "n_unanswerable": len(unanswerable),
        "n_answerable": len(answerable),
        "abstention_accuracy": correct / len(unanswerable) if unanswerable else None,
        "false_abstention_rate": (
            false_abstentions / len(answerable) if answerable else None
        ),
        "n_correct_abstentions": correct,
        "n_false_abstentions": false_abstentions,
    }


# --------------------------------------------------------------------------- #
# Generation metric arithmetic
# --------------------------------------------------------------------------- #
def faithfulness_score(claims: Sequence[dict[str, Any]]) -> float | None:
    """Supported claims / total claims. No claims -> None (not measurable)."""
    if not claims:
        return None
    supported = sum(1 for c in claims if c.get("verdict") == "supported")
    return supported / len(claims)


def has_unsupported_claim(claims: Sequence[dict[str, Any]]) -> bool:
    """True when any claim is contradicted or not found in the context."""
    return any(c.get("verdict") != "supported" for c in claims)


def correctness_score(
    statuses: Sequence[str], contradictions: Sequence[str] | None = None
) -> float | None:
    """Key-point coverage with a contradiction penalty.

    covered=1, partial=0.5, missing=0. Each contradiction subtracts 0.25,
    capped at 0.5, so a contradictory answer cannot beat a merely incomplete one.
    """
    if not statuses:
        return None
    points = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
    coverage = sum(points.get(status, 0.0) for status in statuses) / len(statuses)
    penalty = min(0.5, 0.25 * len(contradictions or []))
    return max(0.0, min(1.0, coverage - penalty))


def citation_score(verdicts: Sequence[dict[str, Any]]) -> float | None:
    """Supported citations / total citations. No citations -> None."""
    if not verdicts:
        return None
    supported = sum(1 for v in verdicts if v.get("supported"))
    return supported / len(verdicts)


def normalize_relevance(score_1_5: int | float) -> float:
    """Map a 1-5 relevance rating to 0-1."""
    value = float(score_1_5)
    return max(0.0, min(1.0, (value - 1.0) / 4.0))


# --------------------------------------------------------------------------- #
# Aggregation and uncertainty
# --------------------------------------------------------------------------- #
def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; caller should pass a non-empty list."""
    return sum(values) / len(values)


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (p in 0..100). Empty -> None."""
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = mean,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap confidence interval for a scalar statistic."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], clean[0]
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(max(1, n_resamples)):
        draw = [clean[rng.randrange(len(clean))] for _ in range(len(clean))]
        samples.append(stat(draw))
    low = percentile(samples, 100.0 * (alpha / 2.0))
    high = percentile(samples, 100.0 * (1.0 - alpha / 2.0))
    return low, high


def summarize_values(
    values: Sequence[float | None],
    *,
    stat: Callable[[Sequence[float]], float] = mean,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Mean (or another statistic) plus a 95% bootstrap CI and sample count."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"value": None, "n": 0, "ci_low": None, "ci_high": None}
    low, high = bootstrap_ci(clean, stat=stat, n_resamples=n_resamples, seed=seed)
    return {
        "value": stat(clean),
        "n": len(clean),
        "ci_low": low,
        "ci_high": high,
    }


def safe_median(values: Sequence[float]) -> float:
    """Median that tolerates an empty sequence by returning 0.0."""
    return statistics.median(values) if values else 0.0
