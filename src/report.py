def render_markdown_summary(pass_rate: float, results: list, drift_info: dict, version: str) -> str:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    lines = [
        f"## Eval run — prompt `{version}`",
        "",
        f"**Pass rate:** {pass_rate:.1%} ({correct}/{total})",
        "",
    ]

    if "reason" in drift_info:
        lines.append(f"_Drift check: {drift_info['reason']}_")
    elif drift_info.get("drift_detected"):
        lines.append(
            f"⚠️ **Drift detected** — recent avg {drift_info['recent_avg']:.1%} vs "
            f"older avg {drift_info['older_avg']:.1%} "
            f"(down {drift_info['drift_magnitude']:.1%})"
        )
    else:
        lines.append(
            f"✅ No drift — recent avg {drift_info['recent_avg']:.1%} vs "
            f"older avg {drift_info['older_avg']:.1%}"
        )
    lines.append("")

    failures = [r for r in results if not r["correct"]]
    if failures:
        lines.append(f"### Failed cases ({len(failures)})")
        lines.append("")
        lines.append("| id | expected | predicted |")
        lines.append("|---|---|---|")
        for r in failures:
            lines.append(f"| {r['id']} | {r['expected']} | {r['predicted']} |")
    else:
        lines.append("All golden cases have passed category classification.")

    return "\n".join(lines)
