# I Revisited the Classic Telco Churn Problem — Here's What Happened When I Combined Graph Data Science with Autonomous Experimentation

A couple of years ago, I spent about 6-7 months working on the telecom customer churn prediction problem — the classic binary classification challenge that every data scientist has seen at some point. Back then, I built graph-based features using Neo4j, experimented with dozens of approaches, and landed on a decent model. It was a solid learning experience, but the process was slow, manual, and honestly exhausting.

Last week, I decided to revisit it. This time, I brought two things I didn't have before: graph data science methodology refined over years of practice, and an autonomous experimentation framework inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) pattern.

The results surprised me.

---

## The Problem (Quick Refresher)

If you haven't worked on it — telecom churn prediction uses customer behavior data (call minutes, service complaints, plan types) to predict who's about to cancel their service. The dataset is small (~3,300 customers), imbalanced (only 14.5% actually churn), and deceptively tricky.

It's deceptive because you can get 85% accuracy by predicting "no churn" for everyone. The real challenge is catching the churners without drowning in false alarms.

## What I Did Differently This Time

### 1. Graph Feature Engineering with Neo4j

The first time around, I treated every customer as an independent row in a table. This time, I modeled them as a graph.

I loaded all 3,333 customers into Neo4j as nodes and created three types of relationships:

- **Geographic proximity** — customers in the same state
- **Area code clusters** — customers sharing the same area code  
- **Behavioral similarity** — customers with similar usage patterns (day/evening/night minutes, service call frequency), connected if their normalized Euclidean distance was below a threshold

This produced a graph with over 33,000 similarity edges. From this graph, I extracted features that a flat table can never give you:

- **State-level churn rate** — what percentage of customers in your state have already churned? (NJ was 28% vs Wyoming at 4%)
- **Neighbor churn rate** — among customers with similar behavior to you, how many have churned? This is essentially a "behavioral contagion" signal.
- **Graph degree** — how many behavioral neighbors do you have? Isolated customers vs. well-connected ones behave differently.

The critical detail: all graph aggregates were computed from training labels only. Test customers got their state's train-computed rates. No data leakage.

### 2. Autonomous Experimentation (The Karpathy Way)

Instead of manually tweaking hyperparameters and running one experiment at a time, I set up an autonomous loop:

```
LOOP:
  1. Edit the experiment code
  2. Commit
  3. Run
  4. Parse metrics
  5. If F1 improved AND metrics balanced → KEEP
  6. Else → DISCARD (git reset)
  7. Repeat
```

The framework has an immutable evaluation harness — the data split, the metrics computation, the output format — none of it changes. The only mutable file is the experiment itself. This separation is what makes autonomous experimentation safe: you can't accidentally game the evaluation.

I ran multiple experiments in parallel, each testing a different strategy:
- XGBoost with graph features
- LightGBM with SMOTE oversampling
- RandomForest with heavy feature engineering
- Stacking ensembles (5 models deep)
- BorderlineSMOTE vs ADASYN vs regular SMOTE
- Learning rate / tree count variations

Some experiments ran in 30 seconds. Some ran for 10+ minutes chasing diminishing returns. A key lesson: **fast, focused experiments that change one variable beat exhaustive grid searches every time.**

### 3. The Features That Actually Mattered

After all the experimentation, the winning model used 32 features. But not all features are created equal. Here's what actually drove the predictions:

**Customer service calls >= 4** was the single most powerful signal. Below 4 calls, churn rate hovers around 10%. At 4 calls, it jumps to 49%. At 5+, it's 55-100%. This is a step function, not a gradual increase. I engineered this as both a binary flag and a squared term to capture both the threshold and the non-linear escalation.

**International plan = Yes** was the second strongest. 40% of international plan holders churned — 3.4x the base rate. The interaction of international plan AND high service calls created the highest-risk segment in the entire dataset.

**Usage ratios** (day minutes as a fraction of total, charge per minute) captured behavioral archetypes that raw numbers miss. A customer using 300 minutes all during the day looks very different from one spreading 300 minutes across all periods.

### 4. What Won (and What Didn't)

**The winning model**: LightGBM + XGBoost ensemble with SMOTE oversampling and out-of-fold threshold tuning.

| Metric | Value |
|--------|-------|
| F1 | **0.935** |
| Precision | **0.989** |
| Recall | **0.887** |

That's 86 out of 97 churners correctly identified, with only 1 false alarm.

**What didn't work**:
- **BorderlineSMOTE and ADASYN** both underperformed regular SMOTE. Tree-based models already handle decision boundaries well — targeted boundary oversampling just added noise.
- **Stacking ensembles** (5 base models + meta-learner) added complexity without beating a simple 2-model average. More models doesn't always mean better.
- **Exhaustive multi-seed grid searches** burned 10+ minutes per run with no improvement. The breakthrough came from SMOTE + lower learning rate — a conceptual change, not a brute-force one.

## The Bigger Takeaway

Two years ago, this took me months. This time, the entire cycle — data exploration, graph construction, feature engineering, 7+ experiments, final model — completed in a single session.

But speed isn't the real lesson. The real lesson is this:

**The best ML improvements come from understanding the problem, not from throwing more compute at it.**

The step-function at 4 customer service calls. The 40% churn rate for international plan holders. The geographic clustering of churn by state. These insights — which came from spending time with the data, not from hyperparameter tuning — are what made the model work.

Graph features added genuine signal that flat tables miss. Behavioral similarity, geographic churn contagion, community structure — these are real phenomena in customer behavior, not just mathematical tricks.

And autonomous experimentation kept me honest. When every experiment is committed, evaluated against fixed metrics, and automatically kept or discarded — there's no room for self-deception about what's actually working.

---

*The full code, detailed report, and experiment history are on GitHub: [AutoChurnResearch](https://github.com/Pratik-Prakash-Sannakki/AutoChurnResearch)*

*Tools used: Python, scikit-learn, XGBoost, LightGBM, Neo4j, imbalanced-learn*

#MachineLearning #DataScience #GraphDataScience #Neo4j #CustomerChurn #AutonomousAI #MLOps
