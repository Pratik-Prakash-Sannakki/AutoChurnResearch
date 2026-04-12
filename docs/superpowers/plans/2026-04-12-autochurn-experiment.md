# AutoChurn Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous churn-classification experiment loop modeled on karpathy/autoresearch — an immutable evaluation harness, a single mutable model file the agent edits, and a program.md the human iterates on, targeting F1 ≥ 0.90 with balanced precision and recall.

**Architecture:** `prepare.py` owns data download, fixed stratified split (cached to disk), and the immutable `evaluate_metrics()` harness. `experiment.py` is the single file the agent edits — it imports from `prepare`, builds a pipeline, and ends with a fixed-format print block. `program.md` gives the agent its loop instructions. All experiments run on the same cached split so results are directly comparable.

**Tech Stack:** Python 3.10+, kagglehub, scikit-learn, xgboost, lightgbm, imbalanced-learn, pandas, numpy, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `autochurn/requirements.txt` | Create | All dependencies |
| `autochurn/.gitignore` | Create | Ignore results.tsv, run.log, __pycache__, .cache |
| `autochurn/prepare.py` | Create | Data download, fixed split, `load_data()`, `evaluate_metrics()` |
| `autochurn/experiment.py` | Create | Baseline DummyClassifier — starting point for agent edits |
| `autochurn/program.md` | Create | Full agent instructions for the autonomous loop |
| `autochurn/tests/test_prepare.py` | Create | Unit tests for evaluate_metrics and output format |

---

## Task 1: Project Scaffold

**Files:**
- Create: `autochurn/requirements.txt`
- Create: `autochurn/.gitignore`

- [ ] **Step 1: Create autochurn directory**

```bash
mkdir -p autochurn/tests
touch autochurn/tests/__init__.py
```

- [ ] **Step 2: Create requirements.txt**

Create `autochurn/requirements.txt`:

```
kagglehub
scikit-learn
xgboost
lightgbm
imbalanced-learn
pandas
numpy
pytest
```

- [ ] **Step 3: Create .gitignore**

Create `autochurn/.gitignore`:

```
results.tsv
run.log
__pycache__/
*.pyc
.cache/
*.pkl
```

- [ ] **Step 4: Install dependencies**

```bash
cd autochurn && pip install -r requirements.txt
```

Expected: All packages install without error.

- [ ] **Step 5: Commit scaffold**

```bash
git add autochurn/requirements.txt autochurn/.gitignore autochurn/tests/__init__.py
git commit -m "feat: autochurn project scaffold"
```

---

## Task 2: Tests for prepare.py (write before implementation)

**Files:**
- Create: `autochurn/tests/test_prepare.py`

- [ ] **Step 1: Write failing tests**

Create `autochurn/tests/test_prepare.py`:

```python
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

    # Exactly at threshold
    p, r = 0.90, 0.75
    balance_ok = "yes" if abs(p - r) <= 0.15 else "no"
    assert balance_ok == "yes"
```

- [ ] **Step 2: Run tests to verify they fail (prepare.py doesn't exist yet)**

```bash
cd autochurn && python -m pytest tests/test_prepare.py -v
```

Expected: `ImportError: No module named 'prepare'` — all 5 tests fail. This confirms TDD red state.

- [ ] **Step 3: Commit failing tests**

```bash
git add autochurn/tests/test_prepare.py
git commit -m "test: failing tests for prepare.py evaluate_metrics"
```

---

## Task 3: Implement prepare.py

**Files:**
- Create: `autochurn/prepare.py`

- [ ] **Step 1: Implement prepare.py**

Create `autochurn/prepare.py`:

```python
"""
AutoChurn — immutable data preparation and evaluation harness.
DO NOT MODIFY — this is the fixed ground truth for all experiments.

Usage (one-time setup):
    cd autochurn && python prepare.py
"""

import os
import pickle

import kagglehub
import numpy as np
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
        with open(SPLIT_CACHE, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    df = _download_and_merge()

    target_col = _find_target_column(df)
    y = df[target_col].map({True: 1, False: 0, "True": 1, "False": 0,
                             "Yes": 1, "No": 0, 1: 1, 0: 0}).astype(int)
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

    cm = confusion_matrix(y_test, y_pred)
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
```

- [ ] **Step 2: Run tests — all must pass**

```bash
cd autochurn && python -m pytest tests/test_prepare.py -v
```

Expected output:
```
tests/test_prepare.py::test_evaluate_metrics_returns_required_keys PASSED
tests/test_prepare.py::test_evaluate_metrics_perfect_predictions PASSED
tests/test_prepare.py::test_evaluate_metrics_all_wrong PASSED
tests/test_prepare.py::test_evaluate_metrics_float_values PASSED
tests/test_prepare.py::test_balance_ok_flag_logic PASSED

5 passed
```

- [ ] **Step 3: Commit**

```bash
git add autochurn/prepare.py
git commit -m "feat: implement prepare.py with load_data and evaluate_metrics"
```

---

## Task 4: Tests for experiment.py output format

**Files:**
- Create: `autochurn/tests/test_experiment_output.py`

- [ ] **Step 1: Write failing tests for output format**

Create `autochurn/tests/test_experiment_output.py`:

```python
"""
Tests that experiment.py produces output the agent can reliably grep.
These tests import and run the experiment output logic directly.
"""
import re
import subprocess
import sys


def test_output_contains_greppable_f1_line():
    """Output must contain a line starting with 'f1:' followed by a float."""
    result = subprocess.run(
        [sys.executable, "experiment.py"],
        capture_output=True, text=True, cwd="."
    )
    output = result.stdout + result.stderr
    match = re.search(r"^f1:\s+\d+\.\d+", output, re.MULTILINE)
    assert match is not None, f"No 'f1:' line found in output:\n{output}"


def test_output_contains_greppable_precision_line():
    """Output must contain a line starting with 'precision:' followed by a float."""
    result = subprocess.run(
        [sys.executable, "experiment.py"],
        capture_output=True, text=True, cwd="."
    )
    output = result.stdout + result.stderr
    match = re.search(r"^precision:\s+\d+\.\d+", output, re.MULTILINE)
    assert match is not None, f"No 'precision:' line found in output:\n{output}"


def test_output_contains_greppable_recall_line():
    """Output must contain a line starting with 'recall:' followed by a float."""
    result = subprocess.run(
        [sys.executable, "experiment.py"],
        capture_output=True, text=True, cwd="."
    )
    output = result.stdout + result.stderr
    match = re.search(r"^recall:\s+\d+\.\d+", output, re.MULTILINE)
    assert match is not None, f"No 'recall:' line found in output:\n{output}"


def test_output_contains_balance_ok_line():
    """Output must contain a line starting with 'balance_ok:' with value 'yes' or 'no'."""
    result = subprocess.run(
        [sys.executable, "experiment.py"],
        capture_output=True, text=True, cwd="."
    )
    output = result.stdout + result.stderr
    match = re.search(r"^balance_ok:\s+(yes|no)", output, re.MULTILINE)
    assert match is not None, f"No 'balance_ok:' line found in output:\n{output}"


def test_output_separator_line():
    """Output must contain the '---' separator line before metrics."""
    result = subprocess.run(
        [sys.executable, "experiment.py"],
        capture_output=True, text=True, cwd="."
    )
    output = result.stdout + result.stderr
    assert "---" in output, f"No '---' separator found in output:\n{output}"
```

- [ ] **Step 2: Run tests — they fail because experiment.py doesn't exist**

```bash
cd autochurn && python -m pytest tests/test_experiment_output.py -v
```

Expected: All 5 tests fail with errors about missing `experiment.py`. Confirms red state.

- [ ] **Step 3: Commit failing tests**

```bash
git add autochurn/tests/test_experiment_output.py
git commit -m "test: failing output format tests for experiment.py"
```

---

## Task 5: Implement baseline experiment.py

**Files:**
- Create: `autochurn/experiment.py`

- [ ] **Step 1: Download data first (required before experiment.py can run)**

```bash
cd autochurn && python prepare.py
```

Expected:
```
Downloading data and creating fixed split...
Split cached to ~/.cache/autochurn/split.pkl
Train: 2666 rows | Test: 667 rows
Churn rate — train: 14.3% | test: 14.2%
Done. Ready to run experiments.
```

- [ ] **Step 2: Implement baseline experiment.py**

Create `autochurn/experiment.py`:

```python
"""
AutoChurn experiment file — the ONE file the agent edits.

Current experiment: Baseline DummyClassifier (stratified random).
This establishes the floor. All future experiments must beat this F1.

Agent: modify EVERYTHING below the imports freely.
Do NOT modify prepare.py.
"""

from prepare import evaluate_metrics, load_data
from sklearn.dummy import DummyClassifier

# ---------------------------------------------------------------------------
# Load data (fixed split — do not change)
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = load_data()

# ---------------------------------------------------------------------------
# Feature engineering
# (none for baseline — agent adds feature engineering here)
# ---------------------------------------------------------------------------

X_train_features = X_train.copy()
X_test_features = X_test.copy()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = DummyClassifier(strategy="stratified", random_state=42)
model.fit(X_train_features, y_train)

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
```

- [ ] **Step 3: Run the output format tests — all must pass**

```bash
cd autochurn && python -m pytest tests/test_experiment_output.py -v
```

Expected:
```
tests/test_experiment_output.py::test_output_contains_greppable_f1_line PASSED
tests/test_experiment_output.py::test_output_contains_greppable_precision_line PASSED
tests/test_experiment_output.py::test_output_contains_greppable_recall_line PASSED
tests/test_experiment_output.py::test_output_contains_balance_ok_line PASSED
tests/test_experiment_output.py::test_output_separator_line PASSED

5 passed
```

- [ ] **Step 4: Run all tests**

```bash
cd autochurn && python -m pytest tests/ -v
```

Expected: All 10 tests pass.

- [ ] **Step 5: Manually verify baseline output**

```bash
cd autochurn && python experiment.py
```

Expected output ends with something like:
```
---
f1:         0.062400
precision:  0.143000
recall:     0.042500
balance_ok: no
```

(Exact numbers will vary — DummyClassifier with stratified random gives near-zero F1 on imbalanced data. `balance_ok: no` is expected for baseline since random predictions won't be balanced.)

- [ ] **Step 6: Commit**

```bash
git add autochurn/experiment.py
git commit -m "feat: baseline experiment.py with DummyClassifier"
```

---

## Task 6: Write program.md

**Files:**
- Create: `autochurn/program.md`

- [ ] **Step 1: Create program.md**

Create `autochurn/program.md`:

````markdown
# AutoChurn

This is an autonomous experiment to improve a binary churn classifier.

## Goal

Improve the churn classifier in `experiment.py` until:
- **F1 ≥ 0.90** on the churn class (positive = 1) — primary metric, drives all keep/discard decisions
- Precision ≥ 0.82 and Recall ≥ 0.82
- `|precision - recall| ≤ 0.15` (balanced — do not sacrifice one for the other)

Once you hit the target, keep going. Push further.

## Setup

To set up a new experiment session:

1. **Agree on a run tag** — propose a tag based on today's date (e.g. `apr12`). Branch `autochurn/<tag>` must not already exist.
2. **Create the branch** — `git checkout -b autochurn/<tag>` from current master.
3. **Read in-scope files** — read these fully before starting:
   - `prepare.py` — fixed evaluation harness. Do not modify.
   - `experiment.py` — the file you edit.
4. **Verify data** — check that `~/.cache/autochurn/split.pkl` exists. If not, run `python prepare.py`.
5. **Initialize results.tsv** — create it with just the header row if it doesn't exist.
6. **Confirm and go.**

## In-scope files

- `prepare.py` — **read only**. Contains `load_data()` and `evaluate_metrics()`. Never modify.
- `experiment.py` — **edit freely**. This is the only file you change.

## What you CAN do in experiment.py

- Feature engineering: encode categoricals, build ratio features, interaction terms, bin continuous variables, correlation-based selection, drop low-signal features
- Handle class imbalance: `class_weight='balanced'`, SMOTE (from `imbalanced-learn`), oversampling, undersampling
- Any sklearn-compatible model: LogisticRegression, RandomForest, XGBoost, LightGBM, SVM, GradientBoosting, ensembles/voting classifiers
- Preprocessing: StandardScaler, MinMaxScaler, imputation, pipelines
- Threshold tuning: find the best decision threshold on the training set (NOT the test set) to maximize F1
- Hyperparameter tuning: GridSearchCV, RandomizedSearchCV, or manual search

## What you CANNOT do

- Modify `prepare.py`
- Change the train/test split or random seed
- Change `evaluate_metrics()` or the fixed print format at the bottom of `experiment.py`
- Install packages not in `requirements.txt`

## Experimentation priorities (rough order — use judgment)

1. **Baseline first** — always Run 0 is `DummyClassifier(strategy="stratified")`. Already in place.
2. **Class imbalance** — high impact. ~14% churn means a naive model ignores the minority class. Try: `class_weight='balanced'`, SMOTE, threshold ≠ 0.5.
3. **Feature engineering** — the dataset has usage stats (day/eve/night minutes, charges, calls) and account features (international plan, voice mail, customer service calls). Good starting points:
   - Encode `International plan` and `Voice mail plan` as binary (1/0)
   - Drop or encode `State` and `Area code` (may be noise)
   - `customer_service_calls ≥ 4` is a strong churn signal — consider a binary flag
   - Total usage = day + eve + night minutes
   - Charge-per-minute ratios (day_charge / day_minutes) — should be near-constant; outliers are signal
4. **Model selection** — escalate: LogisticRegression → RandomForest → XGBoost/LightGBM → ensemble
5. **Threshold tuning** — after fitting, find optimal threshold on training predictions: loop thresholds 0.1–0.9, pick the one that maximizes F1 on the training set, apply to test
6. **Hyperparameter tuning** — once a promising model is found

## Balance rule

Do not accept a keep where `|precision - recall| > 0.15`. A model with precision=0.95, recall=0.60 is not a good outcome even if F1 improved. Rebalance using class weights, threshold adjustment, or resampling before keeping.

## Simplicity criterion

A 0.002 F1 gain that adds 50 lines of complex code is not worth it. Removing complexity while maintaining performance is always a win. Keep `experiment.py` readable.

## Experiment loop

```
LOOP FOREVER:

1. Check git state: current branch and commit
2. Modify experiment.py
3. git commit
4. python experiment.py > run.log 2>&1
5. grep "^f1:\|^precision:\|^recall:\|^balance_ok:" run.log
6. If grep is empty → crash. Run: tail -n 50 run.log to read the error. Fix and retry up to 3 times. If still broken, log crash and revert.
7. Log result to results.tsv
8. If F1 improved AND balance_ok=yes → KEEP (stay on this commit)
9. If F1 equal/worse OR balance_ok=no → DISCARD (git reset --hard <last_kept_commit>)
```

**Timeout:** If a run exceeds 5 minutes, kill it and treat as crash.

**NEVER STOP.** Once the loop begins, do not pause to ask the human. Do not ask "should I continue?". Run indefinitely until manually interrupted.

## Output format (end of every experiment.py run — do not change)

```
---
f1:         0.843200
precision:  0.871000
recall:     0.817400
balance_ok: yes
```

## results.tsv format

Tab-separated. Never commit this file.

```
commit	precision	recall	f1	balance_ok	status	description
```

- `commit` — 7-char git hash
- `precision`, `recall`, `f1` — 6 decimal places
- `balance_ok` — `yes` or `no`
- `status` — `keep`, `discard`, or `crash`
- `description` — brief description of what this experiment tried

Example:
```
commit	precision	recall	f1	balance_ok	status	description
a1b2c3d	0.143000	0.042500	0.062400	no	keep	baseline DummyClassifier stratified
b2c3d4e	0.781000	0.743000	0.761500	yes	keep	LogisticRegression + class_weight balanced
c3d4e5f	0.950000	0.420000	0.582000	no	discard	XGBoost default threshold (recall collapsed)
d4e5f6g	0.823000	0.811000	0.816900	yes	keep	XGBoost + class_weight + customer_service flag
```
````

- [ ] **Step 2: Commit**

```bash
git add autochurn/program.md
git commit -m "feat: add program.md agent instructions"
```

---

## Task 7: End-to-End Verification

**Files:** No new files.

- [ ] **Step 1: Run full test suite**

```bash
cd autochurn && python -m pytest tests/ -v
```

Expected: All 10 tests pass.

- [ ] **Step 2: Simulate what the agent does — run baseline and grep results**

```bash
cd autochurn && python experiment.py > run.log 2>&1
grep "^f1:\|^precision:\|^recall:\|^balance_ok:" run.log
```

Expected output (exact numbers vary):
```
f1:         0.062400
precision:  0.143000
recall:     0.042500
balance_ok: no
```

- [ ] **Step 3: Initialize results.tsv with the baseline entry**

Extract values from above grep output and create `autochurn/results.tsv`:

```
commit	precision	recall	f1	balance_ok	status	description
<7-char-hash>	0.143000	0.042500	0.062400	no	keep	baseline DummyClassifier stratified
```

Replace `<7-char-hash>` with the actual short hash from: `git rev-parse --short HEAD`

- [ ] **Step 4: Verify results.tsv is gitignored**

```bash
cd autochurn && git status
```

Expected: `results.tsv` does NOT appear in git status (it's in `.gitignore`).

- [ ] **Step 5: Final commit**

```bash
git add autochurn/
git commit -m "feat: autochurn experiment framework complete — baseline ready"
```

The framework is ready. Point the agent at `program.md` to begin the autonomous loop:

```
Have a look at autochurn/program.md and let's kick off a new experiment!
```

---

## Summary

| Task | Deliverable |
|------|-------------|
| 1 | `requirements.txt`, `.gitignore`, directory structure |
| 2 | Failing tests for `prepare.py` |
| 3 | `prepare.py` — data download, fixed split, evaluation harness |
| 4 | Failing tests for `experiment.py` output format |
| 5 | `experiment.py` — DummyClassifier baseline |
| 6 | `program.md` — agent loop instructions |
| 7 | End-to-end verified, `results.tsv` initialized with baseline |
