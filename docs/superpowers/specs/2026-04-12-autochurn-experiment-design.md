# AutoChurn Experiment Design

**Date:** 2026-04-12
**Inspired by:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
**Goal:** Autonomously improve a binary churn classifier (telecom dataset) until F1 ≥ 0.90 with balanced precision and recall.

---

## Overview

An AI agent runs a fully autonomous experiment loop — modifying a single Python file, training a classifier, evaluating against a fixed harness, and keeping or discarding changes based on F1 score. Mirrors Karpathy's autoresearch pattern: immutable evaluation, mutable model code, human-owned agent instructions.

---

## Dataset

- **Source:** Kaggle — `mnassrib/telecom-churn-datasets` (downloaded via `kagglehub`)
- **Task:** Binary classification — churn (1) vs. no-churn (0)
- **Size:** ~3,333 rows, ~20 features (call usage stats, account info, customer service calls, plan types)
- **Class distribution:** ~14% churn — imbalanced, so naive accuracy is meaningless
- **Split:** Stratified 80/20 train/test, `random_state=42`, fixed once in `prepare.py` and never changed

---

## File Structure

```
prepare.py      — immutable: data download, fixed split, fixed evaluate_metrics()
experiment.py   — mutable: the ONE file the agent edits (features + model + pipeline)
program.md      — agent instructions: loop behavior, logging, keep/discard rules
results.tsv     — untracked by git, running log of every experiment
run.log         — stdout/stderr of last run (overwritten each experiment)
```

### `prepare.py` (immutable)

Responsibilities:
- Downloads the Kaggle dataset via `kagglehub` (cached after first run)
- Produces a fixed stratified train/test split (`random_state=42`) — saved to disk, reused every run
- Exposes `load_data()` → returns `(X_train, X_test, y_train, y_test)`
- Exposes `evaluate_metrics(model, X_test, y_test)` → returns `{precision, recall, f1}` on the churn class (positive=1)
- Prints confusion matrix and class distribution per run so the agent can see where it's failing
- **Never modified** — this is the ground truth evaluation harness

### `experiment.py` (mutable — agent edits this)

Responsibilities:
- Imports `load_data` and `evaluate_metrics` from `prepare`
- Does all feature engineering, preprocessing, model building, and fitting
- Ends with a fixed-format print block the agent can grep:

```
---
f1:         0.843200
precision:  0.871000
recall:     0.817400
```

The agent can freely change everything inside this file:
- Feature engineering (interactions, ratios, binning, encoding, correlation-based selection)
- Preprocessing (scaling, imputation, resampling — SMOTE, class weights)
- Model choice (any sklearn-compatible: LogisticRegression, RandomForest, XGBoost, SVM, ensembles)
- Hyperparameter tuning (GridSearch, RandomSearch, manual)
- Decision threshold tuning (not locked to 0.5)

### `program.md` (human-owned, iterated between sessions)

The agent reads this file fresh at the start of each session. The human updates it when the agent gets stuck or a new research direction is worth trying. See the Agent Instructions section below for full content.

---

## Metrics

| Metric | Role |
|--------|------|
| F1 (churn class) | **Primary** — drives keep/discard |
| Precision (churn class) | Tracked — logged every run |
| Recall (churn class) | Tracked — logged every run |

**Keep/discard rule:** If F1 improved → keep (advance branch). If F1 equal or worse → discard (`git reset --hard`).

**Balance constraint:** The agent must not accept a keep where `|precision - recall| > 0.15`. A model with precision=0.95, recall=0.60 is not a good outcome even if F1 improved. Both metrics must stay within 15 percentage points of each other.

**Target:** F1 ≥ 0.90, with precision and recall both ≥ 0.82 (minimum for both sides to sustain F1=0.90 with balance).

---

## Baseline

**Run 0 is always the baseline — never skipped.**

Baseline model: `DummyClassifier(strategy="stratified")` from sklearn. Predicts churn/no-churn randomly according to the training class distribution (~14% churn).

Expected baseline performance:
- F1 ≈ 0.05–0.08 (near-zero due to class imbalance)
- Precision ≈ 0.14, Recall ≈ 0.14 (random at class rate)

This is logged as `baseline` in `results.tsv`. All subsequent experiments are measured against this floor.

---

## Experiment Loop

### Setup (once per session)

1. Agent reads `program.md` and all in-scope files (`prepare.py`, `experiment.py`)
2. Proposes a run tag based on today's date (e.g. `apr12`)
3. Creates branch `autochurn/<tag>` — must not already exist
4. Verifies data is downloaded (or runs `python prepare.py`)
5. Initializes `results.tsv` with header row (if it doesn't exist)
6. Confirms setup and begins the loop

### Loop (runs indefinitely)

```
LOOP FOREVER:
1. Check current git state (branch, commit)
2. Modify experiment.py (feature engineering, model, hyperparams)
3. git commit
4. python experiment.py > run.log 2>&1
5. grep "^f1:\|^precision:\|^recall:" run.log
6. If empty → crash. tail -n 50 run.log, attempt fix. Give up after ~3 tries.
7. Log to results.tsv
8. If F1 improved AND balance_ok → keep (stay on commit)
9. If F1 equal/worse OR balance violated → discard (git reset --hard)
```

**Timeout:** If a run exceeds 5 minutes, kill it and treat as crash.

**Crashes:** Fix obvious bugs (typos, missing imports) and re-run. If the idea is fundamentally broken, log `crash`, revert, move on.

**NEVER STOP:** Once the loop begins, the agent does not pause to ask the human. It runs until manually interrupted.

---

## `results.tsv` Format

Tab-separated. NOT comma-separated. Never tracked by git.

```
commit	precision	recall	f1	balance_ok	status	description
```

Example:
```
commit	precision	recall	f1	balance_ok	status	description
a1b2c3d	0.140000	0.140000	0.060000	yes	keep	baseline (DummyClassifier stratified)
b2c3d4e	0.781000	0.743000	0.761500	yes	keep	LogisticRegression + standard scaling
c3d4e5f	0.950000	0.420000	0.582000	no	discard	XGBoost default (recall collapsed)
d4e5f6g	0.823000	0.811000	0.816900	yes	keep	XGBoost + class_weight balanced + feature ratios
```

`balance_ok` = `yes` if `|precision - recall| ≤ 0.15`, else `no`.

---

## Agent Instructions (`program.md` content)

### Goal

Improve the churn classifier in `experiment.py` until F1 ≥ 0.90 on the churn class (positive=1), with precision and recall both ≥ 0.82 and neither more than 15 points apart. The primary metric is F1 — it drives every keep/discard decision.

### In-scope files

- `prepare.py` — read only. Contains the fixed evaluation harness. Do not modify.
- `experiment.py` — edit freely. This is the only file you change.

### Experimentation priorities (rough order)

1. **Baseline first** — always run `DummyClassifier(strategy="stratified")` as Run 0
2. **Class imbalance** — high impact. Try: `class_weight='balanced'`, SMOTE, threshold tuning away from 0.5
3. **Feature engineering** — analyze correlations, build ratios (e.g. day_charge/day_minutes), interaction terms, bin continuous features, encode categoricals properly
4. **Model selection** — escalate: LogisticRegression → RandomForest → XGBoost → LightGBM → ensemble
5. **Hyperparameter tuning** — once a promising model is found, tune it systematically
6. **Threshold tuning** — optimize the decision threshold on the validation set for F1, not just use 0.5

### Balance rule

Do not keep a result where `|precision - recall| > 0.15`. A high-precision, low-recall model is not the goal. Rebalance using class weights, threshold adjustment, or resampling.

### Simplicity criterion

A 0.002 F1 gain that adds 50 lines of complex code is not worth it. Removing complexity while maintaining performance is always a win.

### Keep/discard

- F1 improved AND `|precision - recall| ≤ 0.15` → **keep**
- F1 equal or worse, OR balance violated → **discard** (`git reset --hard`)

### Output format (end of every experiment.py run)

```python
balance_ok = "yes" if abs(precision - recall) <= 0.15 else "no"
print("---")
print(f"f1:         {f1:.6f}")
print(f"precision:  {precision:.6f}")
print(f"recall:     {recall:.6f}")
print(f"balance_ok: {balance_ok}")
```

The agent greps: `grep "^f1:\|^precision:\|^recall:\|^balance_ok:" run.log`

---

## Git Strategy

- Branch: `autochurn/<tag>` (e.g. `autochurn/apr12`)
- Keep = stay on commit, branch advances naturally
- Discard = `git reset --hard <last_kept_commit>`
- `results.tsv` and `run.log` are never committed (add to `.gitignore`)

---

## Dependencies

```
kagglehub
scikit-learn
xgboost
lightgbm
imbalanced-learn   # for SMOTE
pandas
numpy
```

Managed via `requirements.txt`. Agent cannot install packages outside this list.

---

## Success Criteria

The experiment is "complete" when:
- F1 ≥ 0.90 on the held-out test set
- Precision ≥ 0.82 and Recall ≥ 0.82
- `|precision - recall| ≤ 0.15`

The agent does not stop when this is reached — it continues to push further, same as autoresearch.
