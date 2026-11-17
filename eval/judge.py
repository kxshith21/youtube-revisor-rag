"""LLM-as-judge for answer quality, with strict JSON and disk caching.

Rules enforced here:
- temperature 0, strict JSON validated by pydantic, one retry on bad JSON;
- short reasons are stored with every score;
- the judge never sees another metric's score;
- outputs are cached by sha256(prompt + provider + model), so re-runs are free.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

import metrics
from llm import (
    PROMPT_VERSION,
    JsonResult,
    LLMClient,
    LLMJsonError,
    Usage,
    complete_json,
    generator_spec,
    make_client,
    resolve_judge_defaults,
)

# --------------------------------------------------------------------------- #
# JSON schemas returned by the judge
# --------------------------------------------------------------------------- #
class Claim(BaseModel):
    claim: str
    verdict: Literal["supported", "contradicted", "not_found"]
    reason: str = ""


class FaithfulnessOut(BaseModel):
    claims: list[Claim] = Field(default_factory=list)


class RelevanceOut(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = ""


class KeyPointStatus(BaseModel):
    point: str = ""
    status: Literal["covered", "partial", "missing"]
    reason: str = ""


class CorrectnessOut(BaseModel):
    key_points: list[KeyPointStatus] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


class CitationVerdict(BaseModel):
    n: int
    supported: bool
    reason: str = ""


class CitationOut(BaseModel):
    citations: list[CitationVerdict] = Field(default_factory=list)


class RetrievalJudgement(BaseModel):
    relevant: bool
    reason: str = ""


# --------------------------------------------------------------------------- #
# Result wrapper
# --------------------------------------------------------------------------- #
@dataclass
class JudgeResult:
    """One judge call's normalized value, details, reason and usage."""

    value: float | None
    details: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    cached: bool = False


SYSTEM = (
    "You are a strict, literal evaluator for a retrieval-augmented question-answering "
    "system. Follow the task exactly. Use only the information provided in the task; "
    "do not use outside knowledge. Do not reward length, confidence, or style. "
    "Return ONLY a valid JSON object matching the requested schema, with no markdown "
    "fences and no commentary outside the JSON."
)


def _schema_text(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)


class Judge:
    """Provider-agnostic LLM judge with a per-prompt disk cache."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        cache_dir: Path | None = None,
        sleep_seconds: float = 0.0,
        use_cache: bool = True,
        client: LLMClient | None = None,
    ) -> None:
        if client is None:
            self.provider, self.model = resolve_judge_defaults(provider, model)
            self.client = make_client(self.provider, self.model, max_tokens=2048)
        else:
            self.client = client
            self.provider = client.provider
            self.model = client.model
        generator = generator_spec()
        self.same_as_generator = (
            self.provider == generator["provider"] and self.model == generator["model"]
        )
        self.cache_dir = cache_dir or (Path(__file__).resolve().parent / ".cache" / "judge")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_seconds = max(0.0, float(sleep_seconds))
        self.use_cache = use_cache
        self.total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self._calls = 0

    # -- cache / transport -------------------------------------------------- #
    def _cache_path(self, label: str, user: str) -> Path:
        key = hashlib.sha256(
            f"{PROMPT_VERSION}|{self.provider}|{self.model}|{label}|{SYSTEM}|{user}".encode(
                "utf-8"
            )
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _record_usage(self, usage: dict[str, int]) -> None:
        for key in self.total_usage:
            self.total_usage[key] += int(usage.get(key, 0) or 0)

    def _json(
        self,
        label: str,
        user: str,
        schema: type[BaseModel],
    ) -> tuple[JsonResult | None, str | None]:
        """Run one cached JSON judge call; never raises for model/parse failures."""
        path = self._cache_path(label, user)
        if self.use_cache and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                value = schema.model_validate(payload["value"])
                # A cache hit costs no tokens, so record zero usage for it.
                return (
                    JsonResult(
                        value=value,
                        usage=Usage(),
                        raw="",
                        model=payload.get("model", self.model),
                        provider=payload.get("provider", self.provider),
                    ),
                    None,
                )
            except (OSError, ValueError, KeyError) as exc:
                print(f"  judge cache read failed ({exc}); recomputing", file=sys.stderr)

        if self._calls and self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        self._calls += 1
        try:
            result = complete_json(
                self.client,
                SYSTEM,
                user,
                schema,
                temperature=0.0,
                label=f"judge:{label}",
            )
        except (LLMJsonError, Exception) as exc:  # noqa: BLE001 - judge failure is data
            return None, f"{type(exc).__name__}: {exc}"

        usage = result.usage.to_dict()
        self._record_usage(usage)
        if self.use_cache:
            payload = {
                "value": result.value.model_dump(),
                "usage": usage,
                "model": result.model,
                "provider": result.provider,
                "prompt_version": PROMPT_VERSION,
            }
            try:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                tmp.replace(path)
            except OSError as exc:
                print(f"  judge cache write failed ({exc})", file=sys.stderr)
        return result, None

    def _failed(self, error: str) -> JudgeResult:
        return JudgeResult(
            value=None,
            error=error,
            usage=dict(self.total_usage),
        )

    # -- metrics ------------------------------------------------------------ #
    def faithfulness(
        self, question: str, answer: str, context: str
    ) -> JudgeResult:
        """Supported claims / total claims, checked only against retrieved context."""
        if not answer.strip():
            return JudgeResult(value=None, details={"skipped": "empty answer"})
        user = f"""TASK: FAITHFULNESS

Extract every atomic factual claim from the ANSWER. For each claim, classify it using ONLY the CONTEXT:
- "supported": the context states it or directly implies it.
- "contradicted": the context says the opposite.
- "not_found": the context does not contain enough information.
Do not use outside knowledge. Do not judge style, relevance, or correctness against a reference.
If the answer is a refusal such as "I couldn't find this in the indexed videos", return {{"claims": []}}.
A non-refusal answer always contains at least one claim, so never return an empty list for it.

Required JSON schema:
{_schema_text(FaithfulnessOut)}

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""
        result, error = self._json("faithfulness", user, FaithfulnessOut)
        if result is None:
            return self._failed(error or "judge failed")
        out: FaithfulnessOut = result.value  # type: ignore[assignment]
        if not out.claims and not metrics.is_abstention(answer, True):
            # Some judges empty the list on substantive answers. Retry once with
            # an explicit minimum-claims reminder before reporting "not measured".
            retry_user = (
                user
                + "\n\nThe ANSWER above is not a refusal. It contains at least one "
                "factual claim. Extract at least one claim now and never return an "
                'empty `claims` list for it.\n'
            )
            retry, retry_error = self._json(
                "faithfulness:retry", retry_user, FaithfulnessOut
            )
            if retry is None:
                return self._failed(retry_error or error or "judge failed")
            out = retry.value  # type: ignore[assignment]
        claims = [claim.model_dump() for claim in out.claims]
        score = metrics.faithfulness_score(claims)
        unsupported = [c for c in claims if c.get("verdict") != "supported"]
        reason = (
            f"{len(claims) - len(unsupported)}/{len(claims)} claims supported"
            if claims
            else "no atomic claims extracted"
        )
        if unsupported:
            reason += f"; first issue: {unsupported[0].get('reason') or unsupported[0].get('verdict')}"
        return JudgeResult(
            value=score,
            details={"claims": claims, "has_unsupported": metrics.has_unsupported_claim(claims)},
            reason=reason,
            usage=dict(self.total_usage),
            cached=bool(self.use_cache),
        )

    def answer_relevance(self, question: str, answer: str) -> JudgeResult:
        """1-5 rating of how directly the answer addresses the question."""
        if not answer.strip():
            return JudgeResult(value=None, details={"skipped": "empty answer"})
        user = f"""TASK: ANSWER RELEVANCE

Rate how directly the ANSWER addresses the QUESTION on a 1-5 scale:
5 = directly and completely addresses the question.
4 = mostly addresses it, with minor padding or a small gap.
3 = partially addresses it; noticeable padding or drift.
2 = mostly off-topic, evasive, or padded.
1 = does not address the question at all.
A correct refusal such as "I couldn't find this in the indexed videos" is relevant for an unanswerable question and should score 4-5.
Do not judge factual correctness or writing style. Give a one-sentence reason.

Required JSON schema:
{_schema_text(RelevanceOut)}

QUESTION:
{question}

ANSWER:
{answer}
"""
        result, error = self._json("relevance", user, RelevanceOut)
        if result is None:
            return self._failed(error or "judge failed")
        out: RelevanceOut = result.value  # type: ignore[assignment]
        return JudgeResult(
            value=metrics.normalize_relevance(out.score),
            details={"score_1_5": out.score, "reason": out.reason},
            reason=out.reason,
            usage=dict(self.total_usage),
            cached=bool(self.use_cache),
        )

    def correctness(
        self,
        question: str,
        answer: str,
        reference_answer: str,
        key_points: Sequence[str],
    ) -> JudgeResult:
        """Key-point coverage plus contradiction penalty."""
        points = [p for p in key_points if p.strip()]
        if not reference_answer.strip() or not points:
            return JudgeResult(value=None, details={"skipped": "no reference/key points"})
        numbered = "\n".join(f"{i}. {point}" for i, point in enumerate(points, start=1))
        user = f"""TASK: CORRECTNESS

Compare the ANSWER with the REFERENCE ANSWER and the required KEY POINTS.
For every key point, in the same order, decide:
- "covered": the answer states it correctly.
- "partial": the answer mentions it incompletely or imprecisely.
- "missing": the answer omits it or gets it wrong.
Also list any CONTRADICTIONS: statements in the answer that directly conflict with the reference answer or key points.
Use only the reference material; do not use outside knowledge. Do not judge style or length.

Required JSON schema:
{_schema_text(CorrectnessOut)}

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

KEY POINTS:
{numbered}

ANSWER:
{answer}
"""
        result, error = self._json("correctness", user, CorrectnessOut)
        if result is None:
            return self._failed(error or "judge failed")
        out: CorrectnessOut = result.value  # type: ignore[assignment]
        statuses = [status.status for status in out.key_points[: len(points)]]
        statuses.extend(["missing"] * (len(points) - len(statuses)))
        score = metrics.correctness_score(statuses, out.contradictions)
        counts = {name: statuses.count(name) for name in ("covered", "partial", "missing")}
        reason = (
            f"{counts['covered']} covered, {counts['partial']} partial, "
            f"{counts['missing']} missing"
        )
        if out.contradictions:
            reason += f"; {len(out.contradictions)} contradiction(s)"
        return JudgeResult(
            value=score,
            details={
                "key_points": [
                    {
                        "point": points[i],
                        "status": statuses[i],
                        "reason": (
                            out.key_points[i].reason if i < len(out.key_points) else ""
                        ),
                    }
                    for i in range(len(points))
                ],
                "contradictions": out.contradictions,
                "counts": counts,
            },
            reason=reason,
            usage=dict(self.total_usage),
            cached=bool(self.use_cache),
        )

    def citation_accuracy(
        self,
        question: str,
        answer: str,
        citations: Sequence[dict[str, Any]],
    ) -> JudgeResult:
        """For each [n], does source n support the sentence that cites it?"""
        items = list(citations)
        if not items:
            return JudgeResult(value=None, details={"skipped": "no citations"})
        blocks = []
        for item in items:
            blocks.append(
                f"[{item['n']}] SENTENCE: {item.get('sentence', '')}\n"
                f"SOURCE TITLE: {item.get('source_title', '')}\n"
                f"SOURCE TIME: {item.get('source_start_label', '')}\n"
                f"SOURCE TEXT: {item.get('source_text', '')}"
            )
        user = f"""TASK: CITATION ACCURACY

For every citation below, decide whether the cited SOURCE supports the SENTENCE that cites it.
- supported=true only when the source states or directly implies the sentence's claim.
- supported=false when the source is unrelated, contradicts the sentence, or does not contain the claim.
Use only the source text provided. Do not use outside knowledge.

Required JSON schema:
{_schema_text(CitationOut)}

QUESTION:
{question}

ANSWER:
{answer}

CITATIONS:
{chr(10).join(blocks)}
"""
        result, error = self._json("citation", user, CitationOut)
        if result is None:
            return self._failed(error or "judge failed")
        out: CitationOut = result.value  # type: ignore[assignment]
        valid_ns = {int(item["n"]) for item in items}
        by_n = {v.n: v for v in out.citations if v.n in valid_ns}
        verdicts = []
        for item in items:
            n = int(item["n"])
            verdict = by_n.get(n)
            verdicts.append(
                {
                    "n": n,
                    "supported": bool(verdict.supported) if verdict else False,
                    "reason": verdict.reason if verdict else "judge omitted this citation",
                }
            )
        score = metrics.citation_score(verdicts)
        supported = sum(1 for v in verdicts if v["supported"])
        return JudgeResult(
            value=score,
            details={"citations": verdicts},
            reason=f"{supported}/{len(verdicts)} citations supported",
            usage=dict(self.total_usage),
            cached=bool(self.use_cache),
        )

    def retrieval_relevance(
        self,
        question: str,
        reference_answer: str,
        key_points: Sequence[str],
        gold_excerpts: Sequence[str],
        candidate_title: str,
        candidate_text: str,
    ) -> JudgeResult:
        """Would this retrieved chunk help answer the question, even if not gold?"""
        points = "\n".join(f"- {p}" for p in key_points if p.strip()) or "(none)"
        excerpts = "\n\n".join(gold_excerpts) or "(gold excerpts unavailable)"
        user = f"""TASK: RETRIEVAL RELEVANCE

Decide whether the CANDIDATE CHUNK is relevant to answering the QUESTION.
A chunk is relevant if it contains information that helps answer the question, even if it is not one of the gold moments.
It is not relevant if it is about a different topic, only mentions a keyword in passing, or contains no useful information for this question.
Use only the material below. Do not use outside knowledge.

Required JSON schema:
{_schema_text(RetrievalJudgement)}

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

KEY POINTS:
{points}

GOLD EXCERPTS:
{excerpts}

CANDIDATE CHUNK:
TITLE: {candidate_title}
TEXT: {candidate_text}
"""
        result, error = self._json("retrieval", user, RetrievalJudgement)
        if result is None:
            return self._failed(error or "judge failed")
        out: RetrievalJudgement = result.value  # type: ignore[assignment]
        return JudgeResult(
            value=1.0 if out.relevant else 0.0,
            details={"relevant": out.relevant, "reason": out.reason},
            reason=out.reason,
            usage=dict(self.total_usage),
            cached=bool(self.use_cache),
        )
