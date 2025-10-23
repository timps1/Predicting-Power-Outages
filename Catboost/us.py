import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score
from catboost import CatBoostClassifier

# ============== 0. Configs ==============
CSV_PATH = "ALL_USA_TRAINING_DATA.csv"   # <- your dataset
TARGET_COL = "outage_flag"
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Hourly feature prefixes (0..23 hours)
BASE_PREFIXES = [
    "Temp", "Wind Speed", "Wind Direction", "Humidity",
    "Barometer", "RainValue", "HailValue", "ThunderValue"
]

# Columns to remove
DROP_PREFIXES = ["Time_"]
DROP_EXACT = ["date"]

# ============== 1. Load ==============
data = pd.read_csv(CSV_PATH)
print(f"Loaded: {CSV_PATH} | Shape: {data.shape}")

# ============== 2. Basic cleaning ==============
to_drop = []
for c in data.columns:
    if any(c.startswith(p) for p in DROP_PREFIXES) or c in DROP_EXACT:
        to_drop.append(c)
data = data.drop(columns=to_drop, errors="ignore")

obj_cols = [c for c in data.columns if data[c].dtype == "object" and c != TARGET_COL]
if obj_cols:
    print(f"Dropping non-numeric object columns: {obj_cols[:10]}{'...' if len(obj_cols)>10 else ''}")
    data = data.drop(columns=obj_cols, errors="ignore")

if TARGET_COL not in data.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found!")

# ============== 3. Build hourly feature groups ==============
hourly_groups = {}
for prefix in BASE_PREFIXES:
    cols = [c for c in data.columns if c.startswith(prefix + "_")]
    cols = [c for c in cols if c.split("_")[-1].isdigit()]
    cols = sorted(cols, key=lambda x: int(x.split("_")[-1]))
    if len(cols) >= 3:
        hourly_groups[prefix] = cols

print(f"Detected hourly groups: { {k: len(v) for k, v in hourly_groups.items()} }")

# ============== 4. Aggregate & trend features over 24h ==============
aggs = {}
for prefix, cols in hourly_groups.items():
    block = data[cols].astype(float)
    aggs[f"{prefix}_mean24"] = block.mean(axis=1)
    aggs[f"{prefix}_max24"]  = block.max(axis=1)
    aggs[f"{prefix}_min24"]  = block.min(axis=1)
    aggs[f"{prefix}_std24"]  = block.std(axis=1).fillna(0.0)
    aggs[f"{prefix}_delta24"] = block[cols[-1]] - block[cols[0]]
    idx = np.arange(len(cols))
    x = idx.astype(float)
    x = (x - x.mean())
    denom = (x ** 2).sum()
    y = block.values - block.values.mean(axis=1, keepdims=True)
    slope = (y * x).sum(axis=1) / (denom + 1e-12)
    aggs[f"{prefix}_slope24"] = slope

for prefix in ["RainValue", "HailValue", "ThunderValue"]:
    if prefix in hourly_groups:
        cols = hourly_groups[prefix]
        block = data[cols].astype(float)
        aggs[f"{prefix}_sum24"] = block.sum(axis=1)
        aggs[f"{prefix}_any24"] = (block > 0).any(axis=1).astype(int)

agg_df = pd.DataFrame(aggs)
print(f"Built aggregate features: {agg_df.shape[1]} columns")

data_feat = pd.concat([data.drop(columns=[TARGET_COL]), agg_df], axis=1)

# ============== 5. Prepare X, y ==============
X = data_feat
y = data[TARGET_COL].astype(int)

# ============== 6. Train/test split ==============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)

# ============== 7. Imbalance ratio & weights ==============
neg, pos = np.bincount(y_train)
ratio = neg / max(pos, 1)
print(f"Class imbalance ratio (neg/pos) = {ratio:.1f}:1")

sample_weights = np.where(y_train == 1, ratio * 1.5, 1.0)

# ============== 8. CatBoost model (Focal Loss + manual weights) ==============
cat = CatBoostClassifier(
    iterations=3500,
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=4,
    loss_function='Focal:focal_gamma=1.5;focal_alpha=0.5',
    eval_metric='AUC',
    random_seed=RANDOM_SEED,
    verbose=False
)

print("Training CatBoost (Focal + manual weights)...")
cat.fit(X_train, y_train, sample_weight=sample_weights)
cat_probs = cat.predict_proba(X_test)[:, 1]

# ============== 9. Threshold search (maximize F_beta) ==============
def best_threshold_by_beta(y_true, y_prob, beta=2.0):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f_beta = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-12)
    best_idx = np.nanargmax(f_beta)
    thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return thr, precisions[best_idx], recalls[best_idx], f_beta[best_idx], precisions, recalls, thresholds

best_t, p, r, f2, precisions, recalls, thresholds = best_threshold_by_beta(y_test, cat_probs, beta=2.0)
auc = roc_auc_score(y_test, cat_probs)

print("==============================")
print("CatBoost Outage Prediction Results (USA v4)")
print(f"Best threshold (F2) = {best_t:.5f}")
print(f"Precision = {p:.3f} | Recall = {r:.3f} | F2 = {f2:.3f} | AUC = {auc:.3f}")
print("==============================")

print("Beta-Performance Comparison:")
beta_summary = []
for beta in [1.0, 2.0, 3.0]:
    bt, bp, br, bf, _, _, _ = best_threshold_by_beta(y_test, cat_probs, beta=beta)
    beta_summary.append([beta, bt, bp, br, bf])
    print(f"Beta={beta:.0f} → Thr={bt:.5f} | P={bp:.3f} | R={br:.3f} | F{int(beta)}={bf:.3f}")

beta_df = pd.DataFrame(beta_summary, columns=["Beta", "Best_Threshold", "Precision", "Recall", "F_Beta"])

# ============== 10. Plot PR curve ==============
plt.figure(figsize=(6.5, 5.2))
plt.plot(recalls, precisions, linewidth=2, label=f'CatBoost (AUC={auc:.3f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (USA v4)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("usa_catboost_prcurve.png", dpi=300, bbox_inches="tight")
plt.close()
print("PR curve saved to usa_catboost_prcurve.png")

# ============== 11. Save results ==============
summary = pd.DataFrame({
    "Model": ["CatBoost (Focal + USA-agg)"],
    "Best_Threshold_F2": [best_t],
    "Precision": [p],
    "Recall": [r],
    "F2": [f2],
    "AUC": [auc]
})
summary.to_csv("usa_catboost_results.csv", index=False)
beta_df.to_csv("usa_catboost_beta_comparison.csv", index=False)
print("Saved: usa_catboost_results.csv & usa_catboost_beta_comparison.csv")
