import os

import requests

_COLORS = {"pass": "#36a64f", "warning": "#f2c744", "critical": "#e01e5a"}


def send_slack_alert(status: str, pass_rate: float, regressions: list, report_url: str, drift_info: dict) -> None:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        print("SLACK_WEBHOOK_URL not set — skipping Slack alert (this integration is optional).")
        return

    text = f"*Eval Run: {status.upper()}*\nPass rate: {pass_rate:.1%}\nRegressions: {len(regressions)}"
    if drift_info.get("drift_detected"):
        text += f"\n⚠️ Slow drift detected: {drift_info.get('drift_magnitude', 0):.1%} over recent runs"
    if report_url:
        text += f"\n<{report_url}|View full report>"

    response = requests.post(
        webhook,
        json={"attachments": [{"color": _COLORS.get(status, "#999999"), "text": text}]},
        timeout=10,
    )
    response.raise_for_status()
