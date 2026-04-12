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

```python
balance_ok = "yes" if abs(precision - recall) <= 0.15 else "no"
print("---")
print(f"f1:         {f1:.6f}")
print(f"precision:  {precision:.6f}")
print(f"recall:     {recall:.6f}")
print(f"balance_ok: {balance_ok}")
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
