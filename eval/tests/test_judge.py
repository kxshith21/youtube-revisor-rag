"""Judge tests: strict JSON, one parse retry, disk cache, failure isolation."""

from __future__ import annotations

import json
from pathlib import Path

import judge as judge_module
from llm import LLMResponse, Usage


class FakeClient:
    provider = "fake"
    model = "fake-judge-1"

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        json_mode: bool = True,
    ) -> LLMResponse:
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return LLMResponse(
            text=str(item),
            usage=Usage(input_tokens=10, output_tokens=5),
            model=self.model,
            provider=self.provider,
        )


def make_judge(tmp_path: Path, responses: list[object]) -> tuple[object, FakeClient]:
    client = FakeClient(responses)
    judge = judge_module.Judge(
        client=client,
        cache_dir=tmp_path / "cache",
        sleep_seconds=0,
        use_cache=True,
    )
    return judge, client


def test_faithfulness_and_cache_hit(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "claims": [
                {"claim": "a", "verdict": "supported", "reason": "found"},
                {"claim": "b", "verdict": "not_found", "reason": "missing"},
            ]
        }
    )
    judge, client = make_judge(tmp_path, [payload])

    first = judge.faithfulness("q", "answer", "context")
    assert first.value == 0.5
    assert first.error is None
    assert client.calls == 1
    usage_after_first = dict(judge.total_usage)

    second = judge.faithfulness("q", "answer", "context")
    assert second.value == 0.5
    assert client.calls == 1  # served from cache
    assert judge.total_usage == usage_after_first  # cache hits cost nothing


def test_bad_json_retries_once_then_succeeds(tmp_path: Path) -> None:
    good = json.dumps(
        {
            "claims": [
                {"claim": "a", "verdict": "supported", "reason": "found"}
            ]
        }
    )
    judge, client = make_judge(tmp_path, ["not json at all", good])
    result = judge.faithfulness("q", "answer", "context")
    assert result.value == 1.0
    assert client.calls == 2


def test_judge_failure_is_returned_not_raised(tmp_path: Path) -> None:
    judge, client = make_judge(tmp_path, [ValueError("boom")])
    result = judge.faithfulness("q", "answer", "context")
    assert result.value is None
    assert "ValueError" in (result.error or "")
    assert client.calls == 1


def test_empty_answer_skips_the_judge(tmp_path: Path) -> None:
    judge, client = make_judge(tmp_path, [])
    result = judge.faithfulness("q", "   ", "context")
    assert result.value is None
    assert client.calls == 0


def test_relevance_maps_1_to_5_onto_0_to_1(tmp_path: Path) -> None:
    judge, _ = make_judge(
        tmp_path, [json.dumps({"score": 3, "reason": "partly on topic"})]
    )
    result = judge.answer_relevance("q", "answer")
    assert result.value == 0.5
    assert result.details["score_1_5"] == 3


def test_correctness_uses_key_points_and_contradictions(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "key_points": [
                {"point": "a", "status": "covered", "reason": ""},
                {"point": "b", "status": "partial", "reason": ""},
                {"point": "c", "status": "missing", "reason": ""},
            ],
            "contradictions": ["the answer says the opposite of point c"],
        }
    )
    judge, _ = make_judge(tmp_path, [payload])
    result = judge.correctness(
        "q", "answer", "reference", ["a", "b", "c"]
    )
    # (1 + 0.5 + 0) / 3 = 0.5, then -0.25 for one contradiction.
    assert result.value == 0.25
    assert result.details["counts"] == {"covered": 1, "partial": 1, "missing": 1}


def test_correctness_fills_missing_key_point_verdicts(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "key_points": [{"point": "a", "status": "covered", "reason": ""}],
            "contradictions": [],
        }
    )
    judge, _ = make_judge(tmp_path, [payload])
    result = judge.correctness("q", "answer", "reference", ["a", "b"])
    assert result.value == 0.5
    assert [point["status"] for point in result.details["key_points"]] == [
        "covered",
        "missing",
    ]


def test_citation_accuracy_omitted_citation_counts_as_unsupported(
    tmp_path: Path,
) -> None:
    payload = json.dumps(
        {"citations": [{"n": 1, "supported": True, "reason": "ok"}]}
    )
    judge, _ = make_judge(tmp_path, [payload])
    citations = [
        {
            "n": 1,
            "sentence": "A [1].",
            "source_title": "t",
            "source_start_label": "0:00",
            "source_text": "text",
        },
        {
            "n": 2,
            "sentence": "B [2].",
            "source_title": "t",
            "source_start_label": "1:00",
            "source_text": "text",
        },
    ]
    result = judge.citation_accuracy("q", "A [1]. B [2].", citations)
    assert result.value == 0.5
    assert result.details["citations"][1]["supported"] is False


def test_retrieval_relevance(tmp_path: Path) -> None:
    judge, _ = make_judge(
        tmp_path, [json.dumps({"relevant": True, "reason": "helps"})]
    )
    result = judge.retrieval_relevance(
        "q", "reference", ["point"], ["excerpt"], "title", "chunk text"
    )
    assert result.value == 1.0
    assert result.details["relevant"] is True


def test_no_claims_is_not_measurable(tmp_path: Path) -> None:
    judge, _ = make_judge(tmp_path, [json.dumps({"claims": []})])
    result = judge.faithfulness("q", "I couldn't find this.", "context")
    assert result.value is None
    assert "no atomic claims" in result.reason


def test_empty_claims_on_substantive_answer_retries_once(tmp_path: Path) -> None:
    good = json.dumps(
        {
            "claims": [
                {"claim": "a", "verdict": "supported", "reason": "found"}
            ]
        }
    )
    judge, client = make_judge(tmp_path, [json.dumps({"claims": []}), good])
    result = judge.faithfulness(
        "q", "Binary search halves the range [1].", "context"
    )
    assert result.value == 1.0
    assert client.calls == 2
