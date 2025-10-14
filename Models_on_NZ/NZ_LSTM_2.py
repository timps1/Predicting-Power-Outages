import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.utils import class_weight
from sklearn.neighbors import NearestNeighbors
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import warnings, os

# ==================== 环境与随机种子 ====================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ==================== 可调参数 ====================
subset_frac = 0.2                   # 先用 20% 子集加速调参
n_splits = 5
epochs = 5
param = {"lstm_units": 64, "dropout": 0.3, "lr": 1e-3, "batch_size": 32}

# NearMiss: 目标少数/多数比（float）。0.5 表示 少数/多数=0.5 → 多数≈2×少数
nm_ratio_minor_over_major = 0.5

# 自定义 SUS: 目标多数/少数比（float）。2.0 表示 多数≈2×少数（减少得更少）
sus_majority_to_minority_ratio = 2.0
sus_k = 5  # 计算“难样本”时的邻近少数类个数

# focal loss 用于 class_weight 策略（也可换回 BCE）
focal_loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.75)

# ==================== 数据读取与预处理 ====================
df = pd.read_csv("NZ_all_normalised_cleaned.csv")
df = df.sample(frac=subset_frac, random_state=RANDOM_STATE)  # 子集
y = df["outage_flag"].values
X = df.drop(columns=["outage_flag", "date"])

# 强制数值、补 NaN
X = X.apply(pd.to_numeric, errors="coerce").fillna(0).values

# reshape: [样本数, 时间步(24), 每步特征数]
n_steps = 24
n_features = X.shape[1] // n_steps
X = X.reshape(-1, n_steps, n_features)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print("Outage distribution:", np.bincount(y))

# ==================== 自定义 SUS（选择性欠采样） ====================
def selective_undersample_by_distance(X_seq, y_vec, majority_to_minority_ratio=2.0, k=5):
    """
    X_seq: [N, T, F]
    y_vec: [N,]
    思路：仅保留靠近少数类的多数类（“困难样本”）。
    1) 展平到特征空间；
    2) 用少数类做 NearestNeighbors；
    3) 以多数类到少数类的最近距离作为“难度”，距离越小越难；
    4) 选取距离最小的前 top_m 个多数样本，使得
       选后多数 ≈ majority_to_minority_ratio * 少数数目。
    """
    X_flat = X_seq.reshape((X_seq.shape[0], -1))
    classes, counts = np.unique(y_vec, return_counts=True)
    maj_label = classes[np.argmax(counts)]
    min_label = classes[np.argmin(counts)]

    idx_maj = np.where(y_vec == maj_label)[0]
    idx_min = np.where(y_vec == min_label)[0]

    X_min = X_flat[idx_min]
    X_maj = X_flat[idx_maj]

    # 近邻拟合（以少数类为基）
    nn = NearestNeighbors(n_neighbors=min(k, len(idx_min)), algorithm="auto")
    nn.fit(X_min)
    # 对每个多数类，找到到少数类的 k 个最近距离，取“最小距离”作为难度（也可用均值）
    dist, _ = nn.kneighbors(X_maj)
    hardness = dist.min(axis=1)  # 越小越难

    # 选取前 top_m 个“最难”的多数样本
    n_min = len(idx_min)
    n_maj_keep = int(np.clip(majority_to_minority_ratio * n_min, 1, len(idx_maj)))
    top_idx = np.argsort(hardness)[:n_maj_keep]
    keep_majority_idx = idx_maj[top_idx]

    # 合并
    selected = np.concatenate([idx_min, keep_majority_idx])
    np.random.shuffle(selected)

    return X_seq[selected], y_vec[selected]

# ==================== 采样策略列表 ====================
strategies = ["none", "nearmiss", "smote", "class_weight", "sus"]

def build_model():
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        Bidirectional(LSTM(param["lstm_units"], return_sequences=True)),
        Dropout(param["dropout"]),
        Bidirectional(LSTM(param["lstm_units"] // 2)),
        Dropout(param["dropout"]),
        Dense(1, activation="sigmoid")
    ])
    return model

# ==================== 主实验循环 ====================
results_all = []
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

for strategy in strategies:
    print(f"\n=== Strategy: {strategy.upper()} | Params: {param} ===")
    fold_reports = []

    for fold, (tr, va) in enumerate(kf.split(X, y), 1):
        print(f"--- Fold {fold} ---")
        X_train, X_val = X[tr], X[va]
        y_train, y_val = y[tr], y[va]

        class_weights_dict = None

        # ---------- 采样 ----------
        if strategy == "nearmiss":
            nm = NearMiss(version=1, sampling_strategy=nm_ratio_minor_over_major)
            Xtr_flat = X_train.reshape((X_train.shape[0], -1))
            X_res, y_res = nm.fit_resample(Xtr_flat, y_train)
            X_train, y_train = X_res.reshape((-1, n_steps, n_features)), y_res
            print("After NearMiss:", np.bincount(y_train))

        elif strategy == "smote":
            sm = SMOTE(random_state=RANDOM_STATE)
            Xtr_flat = X_train.reshape((X_train.shape[0], -1))
            X_res, y_res = sm.fit_resample(Xtr_flat, y_train)
            X_train, y_train = X_res.reshape((-1, n_steps, n_features)), y_res
            print("After SMOTE:", np.bincount(y_train))

        elif strategy == "class_weight":
            cw = class_weight.compute_class_weight(
                "balanced", classes=np.unique(y_train), y=y_train
            )
            class_weights_dict = dict(enumerate(cw))
            print("Using class weights:", class_weights_dict)

        elif strategy == "sus":
            X_train, y_train = selective_undersample_by_distance(
                X_train, y_train,
                majority_to_minority_ratio=sus_majority_to_minority_ratio,
                k=sus_k
            )
            print("After SUS:", np.bincount(y_train))

        # ---------- 构建/训练模型 ----------
        model = build_model()
        model.compile(
            optimizer=Adam(learning_rate=param["lr"]),
            loss=focal_loss if strategy == "class_weight" else "binary_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=param["batch_size"],
            validation_split=0.1,
            verbose=0,
            class_weight=class_weights_dict if strategy == "class_weight" else None
        )

        # ---------- 阈值调优 ----------
        y_prob = model.predict(X_val, verbose=0).ravel()
        fpr, tpr, thresholds = roc_curve(y_val, y_prob)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        y_pred = (y_prob > best_thresh).astype(int)

        auc = roc_auc_score(y_val, y_prob)
        rep = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        rep["AUC"] = auc
        fold_reports.append(rep)

        print(f"Best threshold: {best_thresh:.3f} | AUC={auc:.4f}")
        K.clear_session()

    # ---------- 平均并记录 ----------
    avg = {}
    for key in fold_reports[0].keys():
        if isinstance(fold_reports[0][key], dict):
            avg[key] = {m: np.mean([r[key][m] for r in fold_reports]) for m in fold_reports[0][key]}
        else:
            avg[key] = np.mean([r[key] for r in fold_reports])

    print(f"\n=== Average Report ({strategy}) ===")
    print(f"Acc={avg['accuracy']:.4f} | Recall_0={avg['0']['recall']:.4f} | "
          f"Recall_1={avg['1']['recall']:.4f} | F1_macro={avg['macro avg']['f1-score']:.4f} | "
          f"AUC={avg['AUC']:.4f}")

    results_all.append({
        "Sampling_Strategy": strategy,
        "Params": str(param),
        "subset_frac": subset_frac,
        "epochs": epochs,
        "nm_ratio_minor_over_major": nm_ratio_minor_over_major if strategy=="nearmiss" else "",
        "sus_majority_to_minority_ratio": sus_majority_to_minority_ratio if strategy=="sus" else "",
        "sus_k": sus_k if strategy=="sus" else "",
        "Accuracy": avg["accuracy"],
        "Recall_0": avg["0"]["recall"],
        "Recall_1": avg["1"]["recall"],
        "F1_macro": avg["macro avg"]["f1-score"],
        "AUC": avg["AUC"]
    })

# ==================== 保存汇总 CSV ====================
out_df = pd.DataFrame(results_all)
out_df.to_csv("sampling_comparison_results.csv", index=False)
print("\n*** All Results Saved to sampling_comparison_results.csv ***")
print(out_df)
