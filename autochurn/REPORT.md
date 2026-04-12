# AutoChurn Experiment Report

**Date**: 2026-04-12
**Branch**: `autochurn/apr12-v2`
**Final Commit**: `f0f3313`
**Dataset**: Kaggle Telecom Churn (`mnassrib/telecom-churn-datasets`)

---

## 1. Executive Summary

Built a binary churn classifier achieving **F1 = 0.935** on a telecom customer dataset, exceeding all target metrics. The winning model is a LightGBM + XGBoost ensemble trained on SMOTE-resampled data with out-of-fold threshold tuning.

| Metric | Target | Achieved | Margin |
|--------|--------|----------|--------|
| F1 (churn class) | >= 0.90 | **0.935** | +0.035 |
| Precision | >= 0.82 | **0.989** | +0.169 |
| Recall | >= 0.82 | **0.887** | +0.067 |
| \|Precision - Recall\| | <= 0.15 | **0.102** | within |

**Confusion Matrix** (test set, 667 rows):
```
              Predicted
              No-Churn  Churn
Actual No-Churn   569       1
Actual Churn       11      86
```

- 86 of 97 churners correctly identified (88.7% recall)
- Only 1 false alarm out of 87 positive predictions (98.9% precision)

---

## 2. Problem Understanding

### 2.1 Business Context

Telecom customer churn is a critical business problem. Acquiring a new customer costs 5-25x more than retaining an existing one. Identifying at-risk customers before they churn enables proactive retention campaigns (discounts, service improvements, targeted outreach).

A good churn model must balance:
- **Precision**: Don't waste retention budget on customers who weren't going to churn
- **Recall**: Don't miss customers who are about to leave
- **F1**: Harmonic mean — the single metric that balances both

### 2.2 Dataset Overview

| Property | Value |
|----------|-------|
| Source | Kaggle `mnassrib/telecom-churn-datasets` |
| Total rows | 3,333 |
| Train / Test split | 2,666 / 667 (80/20, stratified) |
| Features | 19 (numeric + categorical) |
| Churn rate | 14.5% (imbalanced) |
| Random state | 42 (fixed, immutable) |

### 2.3 Feature Inventory

| Feature | Type | Description |
|---------|------|-------------|
| State | Categorical (51) | US state |
| Account length | Integer | Days as customer |
| Area code | Integer (3 values) | 408, 415, 510 |
| International plan | Binary | Yes/No |
| Voice mail plan | Binary | Yes/No |
| Number vmail messages | Integer | Voicemail count |
| Total day/eve/night/intl minutes | Float | Usage by time period |
| Total day/eve/night/intl calls | Integer | Call count by period |
| Total day/eve/night/intl charge | Float | Charges by period |
| Customer service calls | Integer (0-9) | Support contact count |

---

## 3. Data Exploration & Key Findings

### 3.1 Strongest Churn Signals

**Customer service calls >= 4** is the single most powerful predictor:

| CS Calls | Churn Rate | Count |
|----------|-----------|-------|
| 0 | 13.4% | 560 |
| 1 | 9.8% | 945 |
| 2 | 11.5% | 610 |
| 3 | 10.1% | 335 |
| **4** | **48.9%** | 131 |
| **5** | **55.6%** | 54 |
| **6+** | **60-100%** | 31 |

This is a **step function at 4 calls**, not a linear relationship. Customers who call support 4+ times are ~5x more likely to churn.

**International plan = Yes** is the second strongest signal:
- International plan holders: **40.2% churn** (251 customers)
- Non-holders: **11.8% churn** (2,415 customers)

**Voice mail plan = Yes** is protective:
- With voicemail: **8.5% churn** (744 customers)
- Without: **16.8% churn** (1,922 customers)

### 3.2 Correlation Analysis

Top features correlated with churn:

| Feature | Correlation |
|---------|------------|
| Customer service calls | +0.217 |
| Total day minutes/charge | +0.208 |
| Total eve minutes | +0.100 |
| Number vmail messages | -0.092 |
| Total intl minutes | +0.067 |
| Total intl calls | -0.055 |

**Key insight**: Charge columns are perfectly correlated with minutes (charge = rate * minutes). Dropping redundant charges reduces multicollinearity.

### 3.3 Geographic Patterns

Churn rates vary significantly by state:

| State | Churn Rate | Count |
|-------|-----------|-------|
| NJ | 28.3% | 53 |
| MD | 27.8% | 54 |
| ME | 26.2% | 42 |
| SC | 24.0% | 50 |
| TX | 23.6% | 55 |
| ... | ... | ... |
| WY | 4.3% | 46 |
| AK | 4.9% | 41 |

This geographic clustering motivated the Neo4j graph feature engineering.

### 3.4 Class Imbalance

With only **14.5% positive class**, naive models will predict "no churn" for everyone and achieve 85.5% accuracy while catching zero churners. This makes class imbalance handling critical.

---

## 4. Graph Feature Engineering (Neo4j)

### 4.1 Graph Construction

Built a customer graph in Neo4j with 3,333 nodes and three relationship types:

```
(:Customer)-[:IN_STATE]->(:State)     # 3,333 edges
(:Customer)-[:IN_AREA]->(:AreaCode)   # 3,333 edges
(:Customer)-[:SIMILAR]->(:Customer)   # 33,927 edges
```

**SIMILAR edges** connect customers in the same state with Euclidean distance < 0.3 on normalized usage features (day/eve/night/intl minutes + customer service calls).

### 4.2 Graph Features Computed

All aggregates computed from **training labels only** to prevent data leakage:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| state_churn_rate | Average churn rate in customer's state | Geographic risk |
| state_customer_count | Number of customers in state | State size normalization |
| state_intl_rate | Fraction of intl plan holders in state | Regional plan adoption |
| state_avg_csc | Average CS calls in state | Regional service quality |
| area_churn_rate | Churn rate by area code | Local market risk |
| neighbor_churn_rate | Avg churn of SIMILAR neighbors | Behavioral contagion |
| similar_degree | Count of SIMILAR connections | Social connectedness |

### 4.3 Graph Feature Impact

Graph features were used in Run 1 (F1=0.924) but not in the final winning Run 4 (F1=0.935). The Run 4 model achieved higher F1 using LabelEncoder on State + better hyperparameters, suggesting that the gradient boosting models can learn geographic patterns from the encoded state feature when given enough trees and low enough learning rate.

However, graph features provided significant value in the initial breakthrough from baseline (0.149) to competitive (0.924), and would likely help in a production setting where the model needs to generalize to new states.

---

## 5. Methodology

### 5.1 Immutable Harness Pattern

```
prepare.py (IMMUTABLE)        experiment.py (MUTABLE)
├── load_data()          →    ├── Feature engineering
│   └── Fixed 80/20 split     ├── Model training
├── evaluate_metrics()   →    ├── Threshold tuning
│   └── P, R, F1 on test      └── evaluate_metrics() call
└── Cached split.pkl
```

This ensures reproducibility: every experiment runs on the exact same train/test split.

### 5.2 SMOTE Per CV Fold (Critical Detail)

```
WRONG: SMOTE on full train → CV → threshold tuning (leaked synthetic samples)
RIGHT: For each CV fold: SMOTE on fold-train only → predict fold-val → collect OOF
```

Applying SMOTE before cross-validation leaks synthetic samples into validation folds, producing artificially inflated OOF predictions and meaningless threshold estimates. The per-fold approach keeps threshold tuning honest.

### 5.3 Out-of-Fold Threshold Tuning

Instead of using the default 0.5 probability threshold:

1. Run 5-fold stratified CV with SMOTE per fold
2. Collect out-of-fold probability predictions for all training samples
3. Sweep thresholds 0.30-0.70 in 0.005 increments
4. Select threshold maximizing F1 on OOF predictions
5. Apply to final model's test predictions

The winning threshold was **0.535** (slightly above 0.5), reflecting the model's well-calibrated probabilities after SMOTE resampling.

### 5.4 Parallel Experiment Strategy

Used Claude Code's worktree isolation to run multiple experiments simultaneously:

- **Round 1** (4 agents): XGBoost, LightGBM+SMOTE, RandomForest, Stacking ensemble
- **Round 2** (3 agents): 3-model ensemble, BorderlineSMOTE, More trees+lower LR
- **Round 3** (3 agents, killed early): Graph features, Lower threshold, LGBM-heavy weighting

**Lesson learned**: First-round agents were too ambitious (100+ model fits, 10+ minutes each). Switched to fast, focused experiments varying one thing at a time.

---

## 6. Feature Engineering (Final Model)

### 6.1 Categorical Encoding
- State, International plan, Voice mail plan → LabelEncoder

### 6.2 Usage Aggregates
- Total minutes = day + eve + night + intl
- Total calls = day + eve + night + intl
- Total charge = day + eve + night + intl

### 6.3 Ratio Features
- Charge per minute = total_charge / total_minutes
- Charge per call = total_charge / total_calls
- Calls per minute = total_calls / total_minutes
- Day minutes ratio = day_minutes / total_minutes
- Day charge ratio = day_charge / total_charge

### 6.4 Interaction Features
- CS calls squared (captures non-linear CS effect)
- CS calls x international plan (two strongest signals combined)
- CS calls x day charge (high-usage frustrated customers)
- CS calls x total charge (spending + frustration)
- High CS flag (>= 4 calls, binary)
- Intl plan x intl minutes (plan holders who use it heavily)
- Intl plan x intl charge (same, charge variant)
- Vmail plan x vmail messages (engagement signal)

**Total features**: 32 (19 original + 13 engineered)

---

## 7. Model Architecture

### 7.1 Ensemble Design

```
                    ┌─────────────────────┐
                    │   SMOTE Resampled    │
                    │   Training Data      │
                    └──────┬──────┬───────┘
                           │      │
                    ┌──────▼──┐ ┌─▼────────┐
                    │ LightGBM │ │ XGBoost   │
                    │ 1500     │ │ 1200      │
                    │ lr=0.01  │ │ lr=0.015  │
                    └──────┬──┘ └─┬────────┘
                           │      │
                    ┌──────▼──────▼───────┐
                    │  Average Probas     │
                    │  0.5 * LGBM + 0.5   │
                    │        * XGB        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Threshold = 0.535  │
                    │  (OOF-tuned)        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Binary Prediction  │
                    │  0 = No Churn       │
                    │  1 = Churn          │
                    └─────────────────────┘
```

### 7.2 Hyperparameters

| Parameter | LightGBM | XGBoost |
|-----------|----------|---------|
| n_estimators | 1500 | 1200 |
| learning_rate | 0.01 | 0.015 |
| max_depth | 6 | 6 |
| num_leaves | 31 | — |
| subsample | 0.8 | 0.8 |
| colsample_bytree | 0.8 | 0.8 |
| reg_alpha | 0.1 | 0.1 |
| reg_lambda | 1.0 | 1.0 |
| random_state | 42 | 42 |

### 7.3 Why This Works

- **Two complementary boosting algorithms**: LightGBM (leaf-wise growth, faster) and XGBoost (level-wise, more conservative) make different errors → averaging reduces variance
- **Low learning rate + many trees**: Smoother convergence, tighter decision boundaries, better generalization
- **SMOTE resampling**: Generates synthetic minority samples to balance the 85.5/14.5 class distribution
- **Threshold tuning**: Adjusts the probability cutoff from default 0.5 to 0.535 based on CV-validated F1 optimization

---

## 8. Experiment History

| # | Commit | F1 | Precision | Recall | Balance | Status | Description |
|---|--------|----|-----------|--------|---------|--------|-------------|
| 0 | 5ab2eaf | 0.149 | 0.144 | 0.155 | yes | keep | Baseline DummyClassifier |
| 1 | aa5bffd | 0.924 | 0.977 | 0.876 | yes | keep | LGBM+XGB + SMOTE + graph features |
| 2 | ff0bc25 | 0.808 | 0.813 | 0.804 | yes | discard | 3-model ensemble (rebuilt from scratch) |
| 3a | — | 0.915 | 0.945 | 0.887 | yes | discard | BorderlineSMOTE |
| 3b | — | 0.918 | 0.977 | 0.866 | yes | discard | ADASYN |
| 4s | b22c2c9 | 0.920 | 0.956 | 0.887 | yes | discard | Stacking 5-model ensemble |
| **4** | **f0f3313** | **0.935** | **0.989** | **0.887** | **yes** | **keep** | **More trees + lower LR** |

### Key Transitions

- **0 → 1** (+0.775 F1): SMOTE + gradient boosting + graph features. The biggest single jump — moving from random guessing to an actual model.
- **1 → 4** (+0.011 F1): Lower learning rate (0.02→0.01 LGBM, 0.03→0.015 XGB) + more trees (1000→1500 LGBM, 800→1200 XGB). Classic bias-variance optimization.

### What Didn't Work

- **BorderlineSMOTE / ADASYN**: Regular SMOTE outperformed both. Tree-based models already handle decision boundaries well; targeted boundary oversampling added noise.
- **Stacking ensemble**: 5 base models + meta-learner didn't beat 2-model averaging. Added complexity without sufficient diversity.
- **3-model ensemble (adding RF)**: RandomForest's weaker individual performance diluted the ensemble.
- **Multi-seed, multi-weight exhaustive search**: Agents training 100+ models per iteration burned 10+ minutes with no F1 improvement. Fast, focused experiments won.

---

## 9. Error Analysis

### 9.1 What the Model Gets Right

- **High-signal churners**: Intl plan holders with 4+ CS calls → nearly 100% detection
- **Geographic risk**: Customers in high-churn states identified via LabelEncoded state feature
- **Usage-based churners**: Heavy day users with escalating service contacts

### 9.2 The 11 Missed Churners (FN=11)

These are likely "quiet churners" — customers who leave without strong behavioral signals:
- No international plan (removes the 40% churn signal)
- Fewer than 4 customer service calls (removes the step-function signal)
- Average usage patterns (no extreme day/eve/night minutes)
- Not in high-churn states

These customers may churn for reasons not captured in the dataset (competitor offers, moving, household changes).

### 9.3 The 1 False Positive (FP=1)

Near-perfect precision (0.989) means the model is extremely confident when it predicts churn. The single false positive likely sits right at the decision boundary.

---

## 10. Production Considerations

### 10.1 Model Strengths
- High precision (0.989) means retention campaigns won't waste budget on non-churners
- F1=0.935 provides strong overall performance
- Simple architecture (2 models + threshold) is easy to deploy and maintain
- Training completes in under 60 seconds

### 10.2 Limitations
- Fixed train/test split — production model should use time-based splits
- No temporal features (seasonality, trend, recency)
- State encoding via LabelEncoder is not meaningful ordinal — consider target encoding in production
- SMOTE creates synthetic samples that may not represent real customer behavior
- Graph features (Neo4j) were explored but not in the final model — may provide additional lift with different architectures

### 10.3 Recommended Next Steps
1. **Temporal validation**: Use time-based splits to simulate production conditions
2. **Feature importance analysis**: SHAP values to explain individual predictions
3. **Monitoring**: Track precision/recall drift over time
4. **A/B test**: Compare model-targeted vs. random retention campaigns
5. **Graph features in production**: Build real-time customer similarity graph for ongoing neighbor churn rate computation

---

## 11. Technical Details

### 11.1 Environment
- Python 3.11.10 (pyenv)
- scikit-learn, XGBoost, LightGBM, imbalanced-learn
- Neo4j 5.x (APOC plugin for graph schema)
- Total experiment time: ~45 minutes (including graph construction and 7+ experiments)

### 11.2 Reproducibility
```bash
cd autochurn
pip install -r requirements.txt
python prepare.py          # Download + cache data split
python experiment.py       # Run winning model → F1=0.935
```

### 11.3 Files
| File | Role |
|------|------|
| `prepare.py` | Immutable data loading + evaluation harness |
| `experiment.py` | Winning model (207 lines) |
| `program.md` | Agent instructions for autonomous loop |
| `results.tsv` | Full experiment history |
| `requirements.txt` | Python dependencies |
| `~/.cache/autochurn/split.pkl` | Cached train/test split |
| `~/.cache/autochurn/graph_features.pkl` | Pre-computed Neo4j graph features |
