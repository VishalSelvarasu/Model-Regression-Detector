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


def save_run(pass_rate: float, path: str = DEFAULT_HISTORY_PATH, **metadata) -> None:
    history = load_run_history(path)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pass_rate": pass_rate,
    }
    entry.update(metadata)
    history.append(entry)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def check_drift(window: int = 7, threshold: float = 0.05, path: str = DEFAULT_HISTORY_PATH) -> dict:
    """Compare recent vs older pass-rate windows. Read-only."""
    history = load_run_history(path)
    if len(history) < 2 * window:
        return {"drift_detected": False, "reason": f"insufficient history: need {2 * window} runs"}

    recent = [r["pass_rate"] for r in history[-window:]]
    older = history[-2 * window: -window]
    older = [r["pass_rate"] for r in older]

    recent_avg, older_avg = statistics.mean(recent), statistics.mean(older)
    drift = older_avg - recent_avg

    return {
        "drift_detected": drift > threshold,
        "recent_avg": recent_avg,
        "older_avg": older_avg,
        "drift_magnitude": drift,
    }
