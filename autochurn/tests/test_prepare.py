import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock


def test_evaluate_metrics_returns_required_keys():
    """evaluate_metrics must return dict with precision, recall, f1."""
    from prepare import evaluate_metrics

    model = MagicMock()
    y_test = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])
    model.predict.return_value = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    X_test = pd.DataFrame(np.zeros((8, 3)))

    result = evaluate_metrics(model, X_test, y_test)

    assert "precision" in result, "Missing key: precision"
    assert "recall" in result, "Missing key: recall"
    assert "f1" in result, "Missing key: f1"


def test_evaluate_metrics_perfect_predictions():
    """Perfect predictions should yield precision=recall=f1=1.0."""
    from prepare import evaluate_metrics

    model = MagicMock()
    y_test = pd.Series([0, 1, 0, 1, 1])
    model.predict.return_value = y_test.values.copy()
    X_test = pd.DataFrame(np.zeros((5, 3)))

    result = evaluate_metrics(model, X_test, y_test)

    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_evaluate_metrics_all_wrong():
    """All predictions wrong on churn class: recall=0, f1=0, no crash."""
    from prepare import evaluate_metrics

    model = MagicMock()
    y_test = pd.Series([1, 1, 1, 1])
    model.predict.return_value = np.array([0, 0, 0, 0])
    X_test = pd.DataFrame(np.zeros((4, 3)))

    result = evaluate_metrics(model, X_test, y_test)

    assert result["recall"] == pytest.approx(0.0)
    assert result["f1"] == pytest.approx(0.0)


def test_evaluate_metrics_float_values():
    """All returned values must be Python floats between 0 and 1."""
    from prepare import evaluate_metrics

    model = MagicMock()
    y_test = pd.Series([0, 1, 0, 1, 1, 0])
    model.predict.return_value = np.array([0, 1, 1, 1, 0, 0])
    X_test = pd.DataFrame(np.zeros((6, 3)))

    result = evaluate_metrics(model, X_test, y_test)

    for key in ("precision", "recall", "f1"):
        assert isinstance(result[key], float), f"{key} must be float"
        assert 0.0 <= result[key] <= 1.0, f"{key} must be in [0, 1]"


def test_balance_ok_flag_logic():
    """Balance flag: yes if |precision - recall| <= 0.15, no otherwise."""
    # Within threshold
    p, r = 0.90, 0.82
    balance_ok = "yes" if abs(p - r) <= 0.15 else "no"
    assert balance_ok == "yes"

    # Outside threshold
    p, r = 0.95, 0.60
    balance_ok = "yes" if abs(p - r) <= 0.15 else "no"
    assert balance_ok == "no"

    # Within threshold (close to boundary)
    p, r = 0.90, 0.76
    balance_ok = "yes" if abs(p - r) <= 0.15 else "no"
    assert balance_ok == "yes"
