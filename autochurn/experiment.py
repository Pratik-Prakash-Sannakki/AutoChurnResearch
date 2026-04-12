"""
AutoChurn experiment file — the ONE file the agent edits.

Run 1: LightGBM + XGBoost ensemble + SMOTE + graph features + rich features.
Strategy: heavy feature engineering, SMOTE, ensemble of LightGBM + XGBoost
with averaged probabilities, CV-based threshold tuning.

Agent: modify EVERYTHING below the imports freely.
Do NOT modify prepare.py.
"""

import os
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from prepare import evaluate_metrics, load_data
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Load data (fixed split — do not change)
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = load_data()

# ---------------------------------------------------------------------------
# Load graph features and merge by original dataframe index
# ---------------------------------------------------------------------------

GRAPH_PATH = os.path.join(os.path.expanduser("~"), ".cache", "autochurn", "graph_features.pkl")
with open(GRAPH_PATH, "rb") as f:
    graph_features = pickle.load(f)

graph_cols = [
    "state_churn_rate",
    "state_customer_count",
    "state_intl_rate",
    "state_avg_csc",
    "area_churn_rate",
    "area_customer_count",
    "neighbor_churn_rate",
    "similar_degree",
]

gf_indexed = graph_features.set_index("idx")[graph_cols]

X_train_features = X_train.join(gf_indexed, how="left")
X_test_features = X_test.join(gf_indexed, how="left")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to a dataframe."""
    out = df.copy()

    # Encode categoricals as binary
    out["International plan"] = (out["International plan"].str.strip() == "Yes").astype(int)
    out["Voice mail plan"] = (out["Voice mail plan"].str.strip() == "Yes").astype(int)

    # Customer service calls flags (strong churn signal)
    out["csc_high"] = (out["Customer service calls"] >= 4).astype(int)
    out["csc_very_high"] = (out["Customer service calls"] >= 6).astype(int)

    # Total usage across all periods
    out["total_minutes"] = (
        out["Total day minutes"]
        + out["Total eve minutes"]
        + out["Total night minutes"]
        + out["Total intl minutes"]
    )
    out["total_calls"] = (
        out["Total day calls"]
        + out["Total eve calls"]
        + out["Total night calls"]
        + out["Total intl calls"]
    )
    out["total_charge"] = (
        out["Total day charge"]
        + out["Total eve charge"]
        + out["Total night charge"]
        + out["Total intl charge"]
    )

    # Charge-per-minute ratios
    for period in ["day", "eve", "night", "intl"]:
        mins_col = f"Total {period} minutes"
        charge_col = f"Total {period} charge"
        out[f"{period}_cpm"] = np.where(out[mins_col] > 0, out[charge_col] / out[mins_col], 0.0)

    # Interaction: international plan * high day usage
    day_median = out["Total day minutes"].median()
    out["intl_high_day"] = out["International plan"] * (
        out["Total day minutes"] > day_median
    ).astype(int)

    # Interaction: international plan * customer service calls
    out["intl_x_csc"] = out["International plan"] * out["Customer service calls"]

    # Interaction: high CSC * high day minutes
    out["csc_high_x_day"] = out["csc_high"] * out["Total day minutes"]

    # Usage ratios
    out["day_pct"] = np.where(
        out["total_minutes"] > 0,
        out["Total day minutes"] / out["total_minutes"],
        0.0,
    )
    out["intl_pct"] = np.where(
        out["total_minutes"] > 0,
        out["Total intl minutes"] / out["total_minutes"],
        0.0,
    )

    # Voicemail activity
    out["has_voicemail"] = (out["Number vmail messages"] > 0).astype(int)

    # Graph feature interactions
    out["state_churn_x_intl"] = out["state_churn_rate"] * out["International plan"]
    out["state_churn_x_csc"] = out["state_churn_rate"] * out["csc_high"]
    out["area_churn_x_intl"] = out["area_churn_rate"] * out["International plan"]
    out["neighbor_churn_x_csc"] = out["neighbor_churn_rate"] * out["csc_high"]

    # Drop State (replaced by graph features) and redundant charge columns
    drop_cols = [
        "State",
        "Total day charge",
        "Total eve charge",
        "Total night charge",
        "Total intl charge",
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return out


X_train_features = engineer_features(X_train_features)
X_test_features = engineer_features(X_test_features)

print(f"Feature count: {X_train_features.shape[1]}")

# ---------------------------------------------------------------------------
# Threshold tuning via stratified K-fold CV with ensemble
# ---------------------------------------------------------------------------

LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.02,
    "max_depth": 7,
    "num_leaves": 50,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}

XGB_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
    "verbosity": 0,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(X_train_features))

for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_features, y_train)):
    X_tr = X_train_features.iloc[tr_idx]
    y_tr = y_train.iloc[tr_idx]
    X_val = X_train_features.iloc[val_idx]

    # SMOTE on fold training data only
    smote = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

    # LightGBM
    lgbm_fold = LGBMClassifier(**LGBM_PARAMS)
    lgbm_fold.fit(X_tr_sm, y_tr_sm)
    lgbm_proba = lgbm_fold.predict_proba(X_val)[:, 1]

    # XGBoost
    xgb_fold = XGBClassifier(**XGB_PARAMS)
    xgb_fold.fit(X_tr_sm, y_tr_sm)
    xgb_proba = xgb_fold.predict_proba(X_val)[:, 1]

    # Average ensemble
    oof_proba[val_idx] = 0.6 * lgbm_proba + 0.4 * xgb_proba

# Find threshold maximizing F1 on out-of-fold predictions
best_threshold = 0.5
best_f1 = 0.0
for threshold in np.arange(0.10, 0.90, 0.005):
    preds = (oof_proba >= threshold).astype(int)
    score = f1_score(y_train, preds)
    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.3f} (CV F1={best_f1:.4f})")

# ---------------------------------------------------------------------------
# Final model training on full SMOTE'd data
# ---------------------------------------------------------------------------

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_features, y_train)

lgbm = LGBMClassifier(**LGBM_PARAMS)
lgbm.fit(X_train_resampled, y_train_resampled)

xgb = XGBClassifier(**XGB_PARAMS)
xgb.fit(X_train_resampled, y_train_resampled)


# ---------------------------------------------------------------------------
# Wrapper class that uses tuned threshold + ensemble
# ---------------------------------------------------------------------------


class EnsembleThresholdClassifier:
    """Wrapper: averaged LightGBM+XGBoost proba with tuned threshold."""

    def __init__(
        self,
        lgbm_model: LGBMClassifier,
        xgb_model: XGBClassifier,
        threshold: float,
    ) -> None:
        self.lgbm_model = lgbm_model
        self.xgb_model = xgb_model
        self.threshold = threshold

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lgbm_p = self.lgbm_model.predict_proba(X)[:, 1]
        xgb_p = self.xgb_model.predict_proba(X)[:, 1]
        proba = 0.6 * lgbm_p + 0.4 * xgb_p
        return (proba >= self.threshold).astype(int)


model = EnsembleThresholdClassifier(lgbm, xgb, best_threshold)

# ---------------------------------------------------------------------------
# Evaluate — DO NOT CHANGE the print block format below
# ---------------------------------------------------------------------------

metrics = evaluate_metrics(model, X_test_features, y_test)
precision = metrics["precision"]
recall = metrics["recall"]
f1 = metrics["f1"]

balance_ok = "yes" if abs(precision - recall) <= 0.15 else "no"
print("---")
print(f"f1:         {f1:.6f}")
print(f"precision:  {precision:.6f}")
print(f"recall:     {recall:.6f}")
print(f"balance_ok: {balance_ok}")
