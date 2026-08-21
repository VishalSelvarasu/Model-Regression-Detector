import json
import os
from functools import lru_cache

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from src.feature import classify_fault_report
from src.groq_judge import GroqDeepEvalLLM

with open("data/golden_dataset.json", encoding="utf-8") as f:
    golden_cases = json.load(f)
    hard_cases = [c for c in golden_cases if c.get("gate", "hard") == "hard"]
    advisory_cases = [c for c in golden_cases if c.get("gate") == "advisory"]
    known_failures = [c for c in golden_cases if c.get(
        "gate") == "known_failure"]

PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v1")

judge_llm = GroqDeepEvalLLM()

summary_quality = GEval(
    name="SummaryQuality",
    criteria=(
        "Determine whether the actual output identifies the same fault on "
        "the same equipment as the expected output. Differences in wording, "
        "paraphrasing, or omission of minor details such as exact times or "
        "quantities are acceptable and should NOT reduce the score. Mark it "
        "as a failure only when the actual output identifies a different "
        "fault, introduces a problem not mentioned in the input, or clearly "
        "exaggerates or understates the severity of the issue described in "
        "the input."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=judge_llm,
)


@lru_cache(maxsize=None)
def get_prediction(report_text: str):
    return classify_fault_report(report_text, version=PROMPT_VERSION)


@pytest.mark.parametrize("case", hard_cases, ids=[c["id"] for c in hard_cases])
def test_category_classification(case):
    result = get_prediction(case["input"])
    assert result.category == case["expected_category"], (
        f"{case['id']}: expected '{case['expected_category']}', got '{result.category}'"
    )


@pytest.mark.parametrize("case", advisory_cases, ids=[c["id"] for c in advisory_cases])
def test_category_advisory(case):
    result = get_prediction(case["input"])
    assert result.category == case["expected_category"]


@pytest.mark.parametrize("case", golden_cases, ids=[c["id"] for c in golden_cases])
def test_summary_quality(case):
    result = get_prediction(case["input"])
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=result.summary,
        expected_output=case["expected_summary"],
    )
    assert_test(test_case, [summary_quality])


@pytest.mark.xfail(reason="model cannot distinguish resolved faults from active ones; see README", strict=False)
@pytest.mark.parametrize("case", known_failures, ids=[c["id"] for c in known_failures])
def test_category_known_failure(case):
    result = get_prediction(case["input"])
    assert result.category == case["expected_category"]
