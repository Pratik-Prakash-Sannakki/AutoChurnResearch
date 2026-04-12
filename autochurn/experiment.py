"""
AutoChurn experiment file — the ONE file the agent edits.

Run 1: Rich feature engineering + Neo4j graph features + RandomForest
       + class_weight='balanced' + threshold tuning

Graph features via Neo4j (HTTP API):
  - State-level churn rate, intl plan rate, avg day minutes, avg CSC
  - Area-code-level churn rate
  - Behavioral cluster churn rate (same state + CSC bucket)
"""

import json
import warnings

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from prepare import evaluate_metrics, load_data

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Load data (fixed split — do not change)
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = load_data()

# ---------------------------------------------------------------------------
# Neo4j helpers (HTTP API — no Python driver needed)
# ---------------------------------------------------------------------------

NEO4J_URL = "http://127.0.0.1:7474/db/neo4j/tx/commit"
NEO4J_AUTH = ("neo4j", "Pratikps1$")
NEO4J_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


def cypher(query, params=None):
    """Execute a Cypher statement via HTTP API."""
    payload = {"statements": [{"statement": query, "parameters": params or {}}]}
    r = requests.post(NEO4J_URL, auth=NEO4J_AUTH, headers=NEO4J_HEADERS,
                      data=json.dumps(payload), timeout=30)
    resp = r.json()
    if resp.get("errors"):
        raise RuntimeError(f"Neo4j error: {resp['errors']}")
    results = resp.get("results", [])
    if not results or not results[0]["data"]:
        return []
    cols = results[0]["columns"]
    rows = [dict(zip(cols, row["row"])) for row in results[0]["data"]]
    return rows


def build_neo4j_graph_features(X_tr, y_tr, X_te):
    """
    Load training customers into Neo4j, build a state/area-code graph,
    compute aggregated churn-signal features, return for train + test.

    Graph nodes:
      - (:Customer {id, state, area_code, intl_plan, csc, day_min, csc_bucket, churn})
      - (:State {name})
      - (:AreaCode {code})

    Relationships:
      - (Customer)-[:FROM_STATE]->(State)
      - (Customer)-[:IN_AREA]->(AreaCode)
      - (Customer)-[:SIMILAR_TO]->(Customer) where same state + same csc_bucket

    Features extracted (computed on training nodes only):
      - state_churn_rate
      - state_intl_plan_rate
      - state_high_csc_rate   (CSC >= 4)
      - state_avg_day_min
      - state_avg_csc
      - area_churn_rate
      - cluster_churn_rate    (state + csc_bucket neighbors)
    """
    print("[Neo4j] Clearing graph...")
    cypher("MATCH (n) DETACH DELETE n")

    # Build train dataframe with churn labels
    tr = X_tr.copy()
    tr["churn"] = y_tr.values
    tr["csc_bucket"] = pd.cut(tr["Customer service calls"],
                               bins=[-1, 1, 3, 100], labels=["low", "mid", "high"])
    tr["intl_plan_bin"] = (tr["International plan"] == "Yes").astype(int)
    tr["id"] = list(range(len(tr)))

    print(f"[Neo4j] Loading {len(tr)} training customers...")
    # Batch insert customers
    batch_size = 200
    for start in range(0, len(tr), batch_size):
        chunk = tr.iloc[start:start + batch_size]
        params = {"rows": [
            {
                "id": int(row["id"]),
                "state": str(row["State"]),
                "area_code": int(row["Area code"]),
                "intl_plan": int(row["intl_plan_bin"]),
                "csc": int(row["Customer service calls"]),
                "csc_bucket": str(row["csc_bucket"]),
                "day_min": float(row["Total day minutes"]),
                "churn": int(row["churn"]),
            }
            for _, row in chunk.iterrows()
        ]}
        cypher("""
            UNWIND $rows AS r
            CREATE (:Customer {
                id: r.id, state: r.state, area_code: r.area_code,
                intl_plan: r.intl_plan, csc: r.csc, csc_bucket: r.csc_bucket,
                day_min: r.day_min, churn: r.churn
            })
        """, params)

    print("[Neo4j] Creating State and AreaCode nodes + relationships...")
    cypher("""
        MATCH (c:Customer)
        MERGE (s:State {name: c.state})
        MERGE (a:AreaCode {code: c.area_code})
        MERGE (c)-[:FROM_STATE]->(s)
        MERGE (c)-[:IN_AREA]->(a)
    """)

    print("[Neo4j] Querying state-level graph features...")
    state_rows = cypher("""
        MATCH (c:Customer)-[:FROM_STATE]->(s:State)
        WITH s.name AS state,
             count(c) AS n,
             avg(toFloat(c.churn))      AS churn_rate,
             avg(toFloat(c.intl_plan))  AS intl_plan_rate,
             avg(toFloat(c.day_min))    AS avg_day_min,
             avg(toFloat(c.csc))        AS avg_csc,
             sum(CASE WHEN c.csc >= 4 THEN 1 ELSE 0 END) * 1.0 / count(c) AS high_csc_rate
        RETURN state, n, churn_rate, intl_plan_rate, avg_day_min, avg_csc, high_csc_rate
    """)
    state_df = pd.DataFrame(state_rows).set_index("state")

    area_rows = cypher("""
        MATCH (c:Customer)-[:IN_AREA]->(a:AreaCode)
        WITH a.code AS area_code,
             avg(toFloat(c.churn)) AS area_churn_rate
        RETURN area_code, area_churn_rate
    """)
    area_df = pd.DataFrame(area_rows).set_index("area_code")

    print("[Neo4j] Querying behavioral cluster features (state + csc_bucket)...")
    cluster_rows = cypher("""
        MATCH (c:Customer)
        WITH c.state AS state, c.csc_bucket AS bucket,
             avg(toFloat(c.churn)) AS cluster_churn_rate,
             count(c) AS cluster_size
        RETURN state, bucket, cluster_churn_rate, cluster_size
    """)
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df["cluster_key"] = cluster_df["state"] + "_" + cluster_df["bucket"]
    cluster_df = cluster_df.set_index("cluster_key")

    def attach_features(X):
        df = X.copy()
        df["csc_bucket"] = pd.cut(df["Customer service calls"],
                                   bins=[-1, 1, 3, 100], labels=["low", "mid", "high"])
        df["cluster_key"] = df["State"] + "_" + df["csc_bucket"].astype(str)

        # State features
        global_churn_rate = float(state_df["churn_rate"].mean())
        df["state_churn_rate"] = df["State"].map(state_df["churn_rate"]).fillna(global_churn_rate)
        df["state_intl_plan_rate"] = df["State"].map(state_df["intl_plan_rate"]).fillna(state_df["intl_plan_rate"].mean())
        df["state_avg_day_min"] = df["State"].map(state_df["avg_day_min"]).fillna(state_df["avg_day_min"].mean())
        df["state_avg_csc"] = df["State"].map(state_df["avg_csc"]).fillna(state_df["avg_csc"].mean())
        df["state_high_csc_rate"] = df["State"].map(state_df["high_csc_rate"]).fillna(state_df["high_csc_rate"].mean())

        # Area code features
        global_area_churn = float(area_df["area_churn_rate"].mean())
        df["area_churn_rate"] = df["Area code"].map(area_df["area_churn_rate"]).fillna(global_area_churn)

        # Cluster features
        global_cluster_churn = float(cluster_df["cluster_churn_rate"].mean())
        df["cluster_churn_rate"] = df["cluster_key"].map(cluster_df["cluster_churn_rate"]).fillna(global_cluster_churn)

        return df

    tr_feat = attach_features(X_tr)
    te_feat = attach_features(X_te)
    print("[Neo4j] Graph features attached.")
    return tr_feat, te_feat


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(X_tr, y_tr, X_te):
    tr, te = build_neo4j_graph_features(X_tr, y_tr, X_te)

    def transform(df):
        d = pd.DataFrame()
        # Binary encodings
        d["intl_plan"] = (df["International plan"] == "Yes").astype(int)
        d["voicemail_plan"] = (df["Voice mail plan"] == "Yes").astype(int)

        # Customer service calls — most powerful raw feature
        d["csc"] = df["Customer service calls"]
        d["high_csc"] = (df["Customer service calls"] >= 4).astype(int)
        d["very_high_csc"] = (df["Customer service calls"] >= 6).astype(int)

        # Usage features
        d["day_minutes"] = df["Total day minutes"]
        d["day_charge"] = df["Total day charge"]
        d["eve_minutes"] = df["Total eve minutes"]
        d["night_minutes"] = df["Total night minutes"]
        d["intl_minutes"] = df["Total intl minutes"]
        d["intl_calls"] = df["Total intl calls"]
        d["day_calls"] = df["Total day calls"]

        # Total usage
        d["total_minutes"] = df["Total day minutes"] + df["Total eve minutes"] + df["Total night minutes"]
        d["total_charge"] = df["Total day charge"] + df["Total eve charge"] + df["Total night charge"]

        # Interaction: international plan + intl minutes
        d["intl_plan_x_intl_min"] = (df["International plan"] == "Yes").astype(int) * df["Total intl minutes"]

        # Intl calls with plan: intl plan but few calls might be suspicious
        d["intl_plan_low_calls"] = ((df["International plan"] == "Yes") & (df["Total intl calls"] < 3)).astype(int)

        # Voicemail engagement
        d["vmail_messages"] = df["Number vmail messages"]

        # Account length
        d["account_length"] = df["Account length"]

        # High day usage flag (above 250 min is top quartile)
        d["high_day_usage"] = (df["Total day minutes"] > 250).astype(int)

        # State (target encode from neo4j)
        d["state_churn_rate"] = df["state_churn_rate"]
        d["state_high_csc_rate"] = df["state_high_csc_rate"]
        d["state_avg_day_min"] = df["state_avg_day_min"]
        d["state_intl_plan_rate"] = df["state_intl_plan_rate"]

        # Area code churn rate
        d["area_churn_rate"] = df["area_churn_rate"]

        # Behavioral cluster churn rate — most powerful graph feature
        d["cluster_churn_rate"] = df["cluster_churn_rate"]

        # Interaction: personal CSC vs state avg CSC (excess calls above state baseline)
        d["csc_vs_state_avg"] = df["Customer service calls"] - df["state_avg_csc"]

        # Interaction: day minutes vs state avg day minutes
        d["day_min_vs_state_avg"] = df["Total day minutes"] - df["state_avg_day_min"]

        return d

    X_tr_eng = transform(tr)
    X_te_eng = transform(te)
    return X_tr_eng, X_te_eng


X_train_features, X_test_features = engineer_features(X_train, y_train, X_test)

# ---------------------------------------------------------------------------
# Model: RandomForest with class_weight='balanced'
# ---------------------------------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_features, y_train)

# ---------------------------------------------------------------------------
# Threshold tuning on training set (NOT test)
# ---------------------------------------------------------------------------

train_probs = rf.predict_proba(X_train_features)[:, 1]
best_thresh = 0.5
best_f1 = 0.0
for t in np.arange(0.10, 0.90, 0.01):
    preds = (train_probs >= t).astype(int)
    f = f1_score(y_train, preds, pos_label=1, zero_division=0)
    if f > best_f1:
        best_f1 = f
        best_thresh = t

print(f"[Threshold] Best threshold on train: {best_thresh:.2f} (train F1={best_f1:.4f})")


class ThresholdClassifier:
    """Wraps a probabilistic model with a custom decision threshold."""
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        probs = self.model.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)


model = ThresholdClassifier(rf, best_thresh)

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
