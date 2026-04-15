"""
AutoChurn experiment file — the ONE file the agent edits.

Run 4: More trees + lower LR
LGBM(1500, lr=0.01) + XGB(1200, lr=0.015) ensemble with SMOTE + graph features.
Previous best: F1=0.924 with LGBM(1000, lr=0.02) + XGB(800, lr=0.03).

Agent: modify EVERYTHING below the imports freely.
Do NOT modify prepare.py.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from prepare import evaluate_metrics, load_data

# ---------------------------------------------------------------------------
# Load data (fixed split — do not change)
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = load_data()

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features including graph/interaction features."""
    out = df.copy()

    # Encode categoricals
    for col in ["State", "International plan", "Voice mail plan"]:
        if col in out.columns:
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str))

    # Usage aggregates
    out["Total minutes"] = (
        out["Total day minutes"]
        + out["Total eve minutes"]
        + out["Total night minutes"]
        + out["Total intl minutes"]
    )
    out["Total calls"] = (
        out["Total day calls"]
        + out["Total eve calls"]
        + out["Total night calls"]
        + out["Total intl calls"]
    )
    out["Total charge"] = (
        out["Total day charge"]
        + out["Total eve charge"]
        + out["Total night charge"]
        + out["Total intl charge"]
    )

    # Ratios
    out["Charge per minute"] = out["Total charge"] / out["Total minutes"].clip(lower=1e-6)
    out["Charge per call"] = out["Total charge"] / out["Total calls"].clip(lower=1)
    out["Calls per minute"] = out["Total calls"] / out["Total minutes"].clip(lower=1e-6)

    # Day-to-total ratios
    out["Day minutes ratio"] = out["Total day minutes"] / out["Total minutes"].clip(lower=1e-6)
    out["Day charge ratio"] = out["Total day charge"] / out["Total charge"].clip(lower=1e-6)

    # Customer service interaction features (graph-like)
    out["CS calls squared"] = out["Customer service calls"] ** 2
    out["CS calls x intl plan"] = out["Customer service calls"] * out["International plan"]
    out["CS calls x day charge"] = out["Customer service calls"] * out["Total day charge"]
    out["CS calls x total charge"] = out["Customer service calls"] * out["Total charge"]
    out["High CS flag"] = (out["Customer service calls"] >= 4).astype(int)

    # Intl plan interaction features
    out["Intl plan x intl mins"] = out["International plan"] * out["Total intl minutes"]
    out["Intl plan x intl charge"] = out["International plan"] * out["Total intl charge"]

    # Vmail interaction
    out["Vmail plan x vmail msgs"] = out["Voice mail plan"] * out["Number vmail messages"]

    return out


X_train_features = build_features(X_train)
X_test_features = build_features(X_test)

# ---------------------------------------------------------------------------
# SMOTE oversampling on train set
# ---------------------------------------------------------------------------

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_features, y_train)

# ---------------------------------------------------------------------------
# Model — LGBM + XGB ensemble with OOF threshold tuning
# ---------------------------------------------------------------------------

lgbm = LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)

xgb = XGBClassifier(
    n_estimators=1200,
    learning_rate=0.015,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)

# --- OOF threshold tuning on original (non-SMOTE) train data ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(X_train_features))

for fold_train_idx, fold_val_idx in skf.split(X_train_features, y_train):
    X_ft = X_train_features.iloc[fold_train_idx]
    y_ft = y_train.iloc[fold_train_idx]
    X_fv = X_train_features.iloc[fold_val_idx]

    # Apply SMOTE per fold
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_ft_sm, y_ft_sm = sm.fit_resample(X_ft, y_ft)

    lgbm_fold = LGBMClassifier(**lgbm.get_params())
    xgb_fold = XGBClassifier(**xgb.get_params())

    lgbm_fold.fit(X_ft_sm, y_ft_sm)
    xgb_fold.fit(X_ft_sm, y_ft_sm)

    p_lgbm = lgbm_fold.predict_proba(X_fv)[:, 1]
    p_xgb = xgb_fold.predict_proba(X_fv)[:, 1]
    oof_proba[fold_val_idx] = 0.5 * p_lgbm + 0.5 * p_xgb

# Find best threshold by F1 on OOF
best_thr = 0.5
best_f1_oof = 0.0
for thr in np.arange(0.30, 0.70, 0.005):
    preds = (oof_proba >= thr).astype(int)
    from sklearn.metrics import f1_score

    f1_val = f1_score(y_train, preds, pos_label=1)
    if f1_val > best_f1_oof:
        best_f1_oof = f1_val
        best_thr = thr

print(f"OOF best threshold: {best_thr:.3f}  (OOF F1={best_f1_oof:.4f})")

# --- Final models trained on full SMOTE data ---
lgbm.fit(X_train_resampled, y_train_resampled)
xgb.fit(X_train_resampled, y_train_resampled)


# --- Ensemble wrapper that uses threshold ---
class ThresholdEnsemble:
    """Simple averaging ensemble with custom threshold."""

    def __init__(
        self,
        models: list[object],
        threshold: float,
    ) -> None:
        self.models = models
        self.threshold = threshold

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probas = np.mean(
            [m.predict_proba(X)[:, 1] for m in self.models],
            axis=0,
        )
        return (probas >= self.threshold).astype(int)


model = ThresholdEnsemble([lgbm, xgb], threshold=best_thr)

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
