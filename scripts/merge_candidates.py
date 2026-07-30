import argparse
import json
import sys

GOLDEN = "data/golden_dataset.json"
CANDIDATES = "data/golden_dataset_candidates.json"
VALID_CATEGORIES = {"sensor_fault", "communication_error", "mechanical_fault", "nominal"}
REQUIRED_FIELDS = {"id", "input", "expected_category", "expected_summary", "difficulty"}


def main(ids: list | None, merge_all: bool) -> None:
    with open(GOLDEN) as f:
        golden = json.load(f)
    with open(CANDIDATES) as f:
        candidates = json.load(f)

    existing_ids = {c["id"] for c in golden}

    if merge_all:
        selected = candidates
    else:
        wanted = set(ids)
        selected = [c for c in candidates if c["id"] in wanted]
        missing = wanted - {c["id"] for c in selected}
        if missing:
            sys.exit(f"ERROR: ids not found in candidates file: {sorted(missing)}")

    if not selected:
        sys.exit("Nothing selected to merge.")

    for case in selected:
        gaps = REQUIRED_FIELDS - set(case)
        if gaps:
            sys.exit(f"ERROR: {case.get('id', '?')} missing fields: {sorted(gaps)}")
        if case["expected_category"] not in VALID_CATEGORIES:
            sys.exit(
                f"ERROR: {case['id']} has invalid category "
                f"{case['expected_category']!r} (must be one of {sorted(VALID_CATEGORIES)})"
            )
        if case["id"] in existing_ids:
            sys.exit(f"ERROR: {case['id']} already exists in {GOLDEN}")

    golden.extend(selected)
    remaining = [c for c in candidates if c not in selected]

    with open(GOLDEN, "w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)
    with open(CANDIDATES, "w") as f:
        json.dump(remaining, f, indent=2, ensure_ascii=False)

    print(f"Merged {len(selected)} case(s). Golden dataset now has {len(golden)} cases; "
          f"{len(remaining)} candidate(s) remaining.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids", help="Comma-separated case ids to merge, e.g. case_016,case_020")
    group.add_argument("--all", action="store_true", help="Merge all remaining candidates")
    args = parser.parse_args()
    main(args.ids.split(",") if args.ids else None, args.all)
