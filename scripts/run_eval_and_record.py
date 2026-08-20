import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drift import check_drift, save_run  # noqa: E402
from src.feature import classify_fault_report  # noqa: E402
from src.report import render_markdown_summary  # noqa: E402


def run(version: str, dry_run: bool) -> float:
    with open("data/golden_dataset.json", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    errors = 0
    for case in cases:
        try:
            prediction = classify_fault_report(case["input"], version=version)
            predicted = prediction.category
        except Exception as e:
            print(f"  [ERROR] {case['id']}: {e}")
            predicted = "ERROR"
            errors += 1

        correct = predicted == case["expected_category"]
        results.append(
            {
                "id": case["id"],
                "expected": case["expected_category"],
                "predicted": predicted,
                "correct": correct,
            }
        )
        print(
            f"  {'PASS' if correct else 'FAIL'} {case['id']}: expected={case['expected_category']} got={predicted}")

    pass_rate = sum(r["correct"] for r in results) / len(results)
    print(
        f"\nPass rate: {pass_rate:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")

    if errors:
        print(f"{errors} case(s) errored - not recording run")
        sys.exit(1)

    if not dry_run:
        save_run(pass_rate)
        print("Recorded to data/run_history.json")
    else:
        print("--dry-run: not saved to history")

    drift_info = check_drift()  # read-only either way

    os.makedirs("reports", exist_ok=True)
    markdown = render_markdown_summary(pass_rate, results, drift_info, version)
    with open("reports/latest.md", "w", encoding="utf-8") as f:
        f.write(markdown)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")
    else:
        print("\n" + markdown)

    return pass_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", default=os.environ.get("PROMPT_VERSION", "v1"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute pass rate and preview drift without saving to history",
    )
    args = parser.parse_args()
    run(args.version, args.dry_run)
