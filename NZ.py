import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score
from catboost import CatBoostClassifier

# ============== 0. Config ==============
datasets = {
    "AKL": "AKL_Data_normalised_Train_Set.csv",
    "CHCNZ": "CHCNZ_Data_normalised_Train_Set.csv",
    "WLGNZ": "WLGNZ_Data_normalised_Train_Set.csv"
}
TARGET_COL = "outage_flag"
RANDOM_SEED = 42
TEST_SIZE = 0.2

BASE_PREFIXES = [
    "Temp", "Wind Speed", "Wind Direction", "Humidity",
    "Barometer", "RainValue", "HailValue", "ThunderValue"
]
DROP_PREFIXES = ["Time_"]
DROP_EXACT = ["date"]

# ============== 1. Load & merge ==============
frames = []
for name, path in datasets.items():
    df = pd.read_csv(path)
    df["region"] = name
    frames.append(df)
data = pd.concat(frames, ignore_index=True)
print(f"Loaded and merged {len(datasets)} datasets | Shape: {data.shape}")

# ============== 2. Cleaning ==============
to_drop = [c for c in data.columns if any(c.startswith(p) for p in DROP_PREFIXES) or c in DROP_EXACT]
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

# ============== 4. Aggregate & trend features ==============
aggs = {}
for prefix, cols in hourly_groups.items():
    block = data[cols].astype(float)
    aggs[f"{prefix}_mean24"] = block.mean(axis=1)
    aggs[f"{prefix}_max24"]  = block.max(axis=1)
    aggs[f"{prefix}_min24"]  = block.min(axis=1)
    aggs[f"{prefix}_std24"]  = block.std(axis=1).fillna(0.0)
    aggs[f"{prefix}_delta24"] = block[cols[-1]] - block[cols[0]]
    idx = np.arange(len(cols))
    x = (idx - idx.mean())
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

# ===== Extreme & interaction features =====
for pref in ['Wind Speed', 'RainValue', 'ThunderValue', 'Temp', 'Humidity']:
    c = f'{pref}_max24'
    if c in agg_df.columns:
        q95 = agg_df[c].quantile(0.95)
        agg_df[f'{pref}_max24_gt95p'] = (agg_df[c] >= q95).astype(int)

def add_interactions(df):
    add = {}
    if {'Wind Speed_mean24','RainValue_sum24'}.issubset(df.columns):
        add['wind_rain_int'] = df['Wind Speed_mean24'] * (df['RainValue_sum24'] + 1)
    if {'Wind Speed_mean24','Wind Speed_std24'}.issubset(df.columns):
        add['wind_turbulence'] = df['Wind Speed_mean24'] * df['Wind Speed_std24']
    return pd.DataFrame(add)

inter_df = add_interactions(agg_df)
agg_df = pd.concat([agg_df, inter_df], axis=1)
print(f"Built aggregate + extreme + interaction features: {agg_df.shape[1]} columns")

data_feat = pd.concat([data.drop(columns=[TARGET_COL]), agg_df], axis=1)

# ============== 5. Prepare X, y ==============
X = data_feat
y = data[TARGET_COL].astype(int)

# ============== 6. Train/test split ==============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)

# ============== 7. Downsample & weighting ==============
def downsample_negatives(X, y, keep_ratio=0.3, seed=42):
    np.random.seed(seed)
    neg_idx = np.where(y == 0)[0]
    pos_idx = np.where(y == 1)[0]
    keep_neg = np.random.choice(neg_idx, size=int(len(neg_idx) * keep_ratio), replace=False)
    new_idx = np.concatenate([keep_neg, pos_idx])
    return X.iloc[new_idx].reset_index(drop=True), y.iloc[new_idx].reset_index(drop=True)

X_train_ds, y_train_ds = downsample_negatives(X_train, y_train, keep_ratio=0.3, seed=RANDOM_SEED)
neg, pos = np.bincount(y_train_ds)
ratio = neg / max(pos, 1)
print(f"Class imbalance ratio after downsampling (neg/pos) = {ratio:.1f}:1")

sample_weights = np.where(y_train_ds == 1, ratio * 2.5, 1.0)

# ============== 8. CatBoost model ==============
cat = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.015,
    depth=10,
    l2_leaf_reg=6,
    loss_function='Focal:focal_gamma=2.0;focal_alpha=0.6',
    eval_metric='AUC',
    subsample=0.7,
    rsm=0.7,
    random_seed=RANDOM_SEED,
    verbose=False
)

print("Training CatBoost (Enhanced High-Recall Mode)...")
cat.fit(X_train_ds, y_train_ds, sample_weight=sample_weights)
cat_probs = cat.predict_proba(X_test)[:, 1]

# ============== 9. Target recall threshold ==============
def threshold_for_target_recall(y_true, y_prob, target_recall=0.60):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    idx = np.where(r >= target_recall)[0]
    if len(idx) == 0:
        best = np.argmax(r)
        use_thr = thr[min(best, len(thr)-1)] if len(thr) else 0.5
        return use_thr, p[best], r[best]
    j = idx[-1]
    use_thr = thr[min(j, len(thr)-1)] if len(thr) else 0.5
    return use_thr, p[j], r[j]

best_thr, p, r = threshold_for_target_recall(y_test, cat_probs, target_recall=0.60)
auc = roc_auc_score(y_test, cat_probs)
print("==============================")
print("CatBoost Outage Prediction Results (NZ Combined Enhanced)")
print(f"Target Recall = 0.60 → Threshold = {best_thr:.5f}")
print(f"Precision = {p:.3f} | Recall = {r:.3f} | AUC = {auc:.3f}")
print("==============================")

# ============== 10. Plot PR Curve ==============
precisions, recalls, thresholds = precision_recall_curve(y_test, cat_probs)
plt.figure(figsize=(6.5, 5.2))
plt.plot(recalls, precisions, linewidth=2, label=f'CatBoost (AUC={auc:.3f})')
plt.scatter(r, p, color='red', label=f'Target Recall {r:.2f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (NZ Combined Enhanced)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("nz_enhanced_catboost_prcurve.png", dpi=300, bbox_inches="tight")
plt.close()
print("PR curve saved to nz_enhanced_catboost_prcurve.png")

# ============== 11. Save results ==============
summary = pd.DataFrame({
    "Model": ["CatBoost (Enhanced + Downsample + Focal)"],
    "Target_Recall": [r],
    "Precision": [p],
    "Recall": [r],
    "Threshold": [best_thr],
    "AUC": [auc]
})
summary.to_csv("nz_enhanced_catboost_results.csv", index=False)
print("Saved: nz_enhanced_catboost_results.csv")
