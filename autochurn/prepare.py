"""
AutoChurn — immutable data preparation and evaluation harness.
DO NOT MODIFY — this is the fixed ground truth for all experiments.

Usage (one-time setup):
    cd autochurn && python prepare.py
"""

import os
import pickle

import kagglehub
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants (fixed — do not change)
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autochurn")
SPLIT_CACHE = os.path.join(CACHE_DIR, "split.pkl")
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _download_and_merge() -> pd.DataFrame:
    """Download dataset via kagglehub and merge all CSV files into one DataFrame."""
    path = kagglehub.dataset_download("mnassrib/telecom-churn-datasets")
    csv_files = []
    for root, _, files in os.walk(path):
        for f in sorted(files):
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {path}")
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def _find_target_column(df: pd.DataFrame) -> str:
    """Find the churn target column regardless of exact naming."""
    for candidate in ["Churn", "churn", "CHURN"]:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Could not find churn column. Available: {list(df.columns)}")


def load_data():
    """
    Load fixed stratified train/test split.
    Downloads and caches data on first call. Returns cached split on subsequent calls.

    Returns:
        (X_train, X_test, y_train, y_test) — DataFrames/Series with original dtypes.
    """
    if os.path.exists(SPLIT_CACHE):
        try:
            with open(SPLIT_CACHE, "rb") as f:
                split = pickle.load(f)
            if not (isinstance(split, tuple) and len(split) == 4):
                raise ValueError("Unexpected cache format")
            return split
        except Exception as e:
            print(f"Warning: cache invalid ({e}), regenerating...")
            os.remove(SPLIT_CACHE)

    os.makedirs(CACHE_DIR, exist_ok=True)
    df = _download_and_merge()

    target_col = _find_target_column(df)
    y = df[target_col].map({True: 1, False: 0, "True": 1, "False": 0,
                             "Yes": 1, "No": 0, 1: 1, 0: 0})
    if y.isna().any():
        bad = df[target_col][y.isna()].unique().tolist()
        raise ValueError(f"Unmapped churn values after label encoding: {bad}")
    y = y.astype(int)
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    split = (X_train, X_test, y_train, y_test)
    with open(SPLIT_CACHE, "wb") as f:
        pickle.dump(split, f)

    print(f"Split cached to {SPLIT_CACHE}")
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Churn rate — train: {y_train.mean():.1%} | test: {y_test.mean():.1%}")

    return split


# ---------------------------------------------------------------------------
# Evaluation harness (DO NOT CHANGE — fixed ground truth)
# ---------------------------------------------------------------------------

def evaluate_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate model on the fixed test set.

    Returns dict with keys: precision, recall, f1 (all float, churn class = 1).
    Also prints confusion matrix and class distribution for agent diagnostics.
    """
    y_pred = model.predict(X_test)

    precision = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"Confusion matrix (rows=actual, cols=predicted):")
    print(f"  TN={cm[0, 0]:4d}  FP={cm[0, 1]:4d}")
    print(f"  FN={cm[1, 0]:4d}  TP={cm[1, 1]:4d}")
    dist = dict(y_test.value_counts().sort_index())
    print(f"Class distribution (test): no-churn={dist.get(0, 0)}  churn={dist.get(1, 0)}")

    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Main — one-time setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Downloading data and creating fixed split...")
    X_train, X_test, y_train, y_test = load_data()
    print("Done. Ready to run experiments.")
