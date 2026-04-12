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
