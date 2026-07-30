import json
import os

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from src.feature import classify_fault_report
from src.groq_judge import GroqDeepEvalLLM

with open("data/golden_dataset.json") as f:
    golden_cases = json.load(f)

PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v1")

judge_llm = GroqDeepEvalLLM()

summary_quality = GEval(
    name="SummaryQuality",
    criteria=(
        "Determine whether 'actual output' describes the same core fault or "
        "status as 'expected output', allowing for different wording, as "
        "long as it doesn't invent details unsupported by 'input'."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
    model=judge_llm,
)


@pytest.mark.parametrize("case", golden_cases, ids=[c["id"] for c in golden_cases])
def test_fault_report_classification(case):
    result = classify_fault_report(case["input"], version=PROMPT_VERSION)

    assert result.category == case["expected_category"], (
        f"{case['id']}: expected category '{case['expected_category']}', "
        f"got '{result.category}' for input: {case['input']!r}"
    )

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=result.summary,
        expected_output=case["expected_summary"],
    )
    assert_test(test_case, [summary_quality])
