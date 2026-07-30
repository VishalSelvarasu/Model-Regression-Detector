import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_HISTORY_PATH = "data/run_history.json"


def load_run_history(path: str = DEFAULT_HISTORY_PATH) -> list:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_run(pass_rate: float, path: str = DEFAULT_HISTORY_PATH) -> None:
    history = load_run_history(path)
    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pass_rate": pass_rate,
        }
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def check_drift(window: int = 7, threshold: float = 0.05, path: str = DEFAULT_HISTORY_PATH) -> dict:
    """Compare recent vs older pass-rate windows. Read-only."""
    history = load_run_history(path)
    if len(history) < window:
        return {"drift_detected": False, "reason": "insufficient history"}

    recent = [r["pass_rate"] for r in history[-window:]]
    older = history[-2 * window: -window]
    older = [r["pass_rate"] for r in older] if len(older) == window else recent

    recent_avg, older_avg = statistics.mean(recent), statistics.mean(older)
    drift = older_avg - recent_avg

    return {
        "drift_detected": drift > threshold,
        "recent_avg": recent_avg,
        "older_avg": older_avg,
        "drift_magnitude": drift,
    }
