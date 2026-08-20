import json

import pytest

from src.drift import check_drift, load_run_history, save_run


def _write_history(path, pass_rates):
    history = [{"pass_rate": rate} for rate in pass_rates]
    path.write_text(json.dumps(history), encoding="utf-8")


def test_empty_history_is_insufficient(tmp_path):
    p = tmp_path / "history.json"
    p.write_text("[]", encoding="utf-8")

    result = check_drift(window=7, path=str(p))

    assert result["drift_detected"] is False
    assert "insufficient" in result["reason"].lower()


def test_thirteen_entries_is_still_insufficient(tmp_path):
    p = tmp_path / "history.json"
    _write_history(p, [1.0] * 6 + [0.8] * 7)

    result = check_drift(window=7, path=str(p))

    assert result["drift_detected"] is False
    assert "insufficient" in result["reason"].lower()


def test_fourteen_entries_with_drop_detects_drift(tmp_path):
    p = tmp_path / "history.json"
    _write_history(p, [1.0] * 7 + [0.8] * 7)

    result = check_drift(window=7, path=str(p))

    assert result["drift_detected"] is True
    assert result["drift_magnitude"] == pytest.approx(0.2)


def test_fourteen_identical_entries_has_no_drift(tmp_path):
    p = tmp_path / "history.json"
    _write_history(p, [1.0] * 14)

    result = check_drift(window=7, path=str(p))

    assert result["drift_detected"] is False


def test_save_run_then_load_run_history(tmp_path):
    p = tmp_path / "history.json"

    save_run(0.9, path=str(p))
    history = load_run_history(path=str(p))

    assert len(history) == 1
    assert history[0]["pass_rate"] == pytest.approx(0.9)
    assert "timestamp" in history[0]
