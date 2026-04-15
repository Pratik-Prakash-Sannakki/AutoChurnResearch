# AutoChurnResearch

Autonomous experiment framework for building a high-performance telecom churn classifier, inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## Results

| Metric | Target | Achieved |
|--------|--------|----------|
| F1 (churn class) | >= 0.90 | **0.935** |
| Precision | >= 0.82 | **0.989** |
| Recall | >= 0.82 | **0.887** |
| \|Precision - Recall\| | <= 0.15 | **0.102** |

**Confusion Matrix** (test set, 667 rows, 97 churners):
```
              Predicted
              No-Churn  Churn
Actual No-Churn   569       1
Actual Churn       11      86
```

## Winning Model

LightGBM (1500 trees, lr=0.01) + XGBoost (1200 trees, lr=0.015) ensemble with:
- SMOTE oversampling per CV fold (no data leakage)
- 32 features (19 original + 13 engineered interaction/ratio features)
- Out-of-fold threshold tuning (threshold=0.535)

## Architecture

```
prepare.py (IMMUTABLE)         experiment.py (MUTABLE)
├── load_data()           →    ├── Feature engineering (32 features)
│   └── Fixed 80/20 split      ├── SMOTE per CV fold
├── evaluate_metrics()    →    ├── LGBM + XGB ensemble
│   └── P, R, F1 on test       ├── OOF threshold tuning
└── Cached split.pkl            └── ThresholdEnsemble wrapper
```

- `prepare.py` — Immutable data loading and evaluation harness. Never modified.
- `experiment.py` — The single mutable file. Contains all feature engineering, model training, and threshold tuning.
- `program.md` — Agent instructions for the autonomous experiment loop.

## Dataset

[Kaggle Telecom Churn](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets) — 3,333 customers, 19 features, 14.5% churn rate.

Key predictors discovered:
- **Customer service calls >= 4**: Step-function from ~10% to ~50% churn rate
- **International plan = Yes**: 40.2% churn (3.4x base rate)
- **Voice mail plan = Yes**: Protective (8.5% vs 16.8% churn)

## Graph Features (Neo4j)

Customer similarity graph built in Neo4j using the following graph data science methods:

**Graph Construction**:
- **Hub-and-spoke pattern**: State and AreaCode hub nodes with `IN_STATE` / `IN_AREA` relationships (3,333 edges each)
- **k-Nearest Neighbors (kNN) similarity**: Customers connected via `SIMILAR` edges based on Euclidean distance on normalized usage features (day/eve/night/intl minutes + CS calls). Threshold: distance < 0.3. Produced 33,927 behavioral similarity edges.

**Graph Feature Extraction Methods**:
| Feature | Method | Description |
|---------|--------|-------------|
| state_churn_rate | **Node aggregation via hub** | Mean churn label of all training customers connected to the same State hub node |
| area_churn_rate | **Node aggregation via hub** | Same as above for AreaCode hub |
| neighbor_churn_rate | **Neighborhood aggregation** | Mean churn label of all SIMILAR-connected training neighbors (1-hop traversal) |
| similar_degree | **Degree centrality** | Count of SIMILAR edges per customer — measures behavioral connectedness |
| state_intl_rate | **Hub-level feature propagation** | Fraction of international plan holders in the customer's state |
| state_avg_csc | **Hub-level feature propagation** | Average customer service calls in the customer's state |
| state_customer_count | **Hub cardinality** | Number of customers per state — normalizes small-sample states |

**Leakage prevention**: All aggregates computed from training labels only. Test customers receive their state/area's train-computed rates.

Graph features drove the initial breakthrough (F1: 0.149 -> 0.924) but the final model achieved 0.935 without them via better hyperparameter tuning.

## Experiment History

| Run | F1 | Strategy | Status |
|-----|-----|----------|--------|
| 0 | 0.149 | Baseline DummyClassifier | keep |
| 1 | 0.924 | LGBM+XGB + SMOTE + graph features | keep |
| 2 | 0.808 | 3-model ensemble (LGBM+XGB+RF) | discard |
| 3a | 0.915 | BorderlineSMOTE | discard |
| 3b | 0.918 | ADASYN | discard |
| 4s | 0.920 | Stacking 5-model ensemble | discard |
| **4** | **0.935** | **More trees + lower LR** | **keep** |

## Detailed Report

For the full writeup including data exploration, feature engineering rationale, model architecture, error analysis, and production considerations, see [autochurn/REPORT.md](autochurn/REPORT.md).

## Quick Start

```bash
cd autochurn
pip install -r requirements.txt
python prepare.py          # Download + cache data split
python experiment.py       # Run winning model -> F1=0.935
```

## Tech Stack

- Python 3.11, scikit-learn, XGBoost, LightGBM, imbalanced-learn
- Neo4j (graph feature engineering)
- pandas, numpy, pytest
