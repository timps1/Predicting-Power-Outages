import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import warnings, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# ========== 读取数据 ==========
df = pd.read_csv("ALL_USA_TRAINING_DATA.csv")
df = df.sample(frac=0.2, random_state=42)  # 采样子集测试

y = df["outage_flag"].values
X = df.drop(columns=["outage_flag", "date"])

# 转数值，避免 object 类型
X = X.apply(pd.to_numeric, errors="coerce").fillna(0).values

# reshape → [样本数, 时间步, 特征数]
n_steps = 24
n_features = X.shape[1] // n_steps
X = X.reshape(-1, n_steps, n_features)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print("Outage distribution:", np.bincount(y))

# ========== 超参数 ==========
params = {"lstm_units": 64, "dropout": 0.3, "lr": 1e-3, "batch_size": 32}
results = []

# Focal Loss 可替换为加权 BCE，这里保留 focal
focal_loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.75)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
reports = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n--- Fold {fold} ---")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # class_weight
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw = dict(enumerate(cw))
    print("Class weights:", cw)

    # LSTM 模型
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        Bidirectional(LSTM(params["lstm_units"], return_sequences=True)),
        Dropout(params["dropout"]),
        Bidirectional(LSTM(params["lstm_units"] // 2)),
        Dropout(params["dropout"]),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=params["lr"]),
        loss=focal_loss,
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=params["batch_size"],
        validation_split=0.1,
        verbose=0,
        class_weight=cw
    )

    # -------- 阈值调优（基于 ROC 曲线）--------
    y_prob = model.predict(X_val, verbose=0).ravel()
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    best_thresh = thresholds[np.argmax(tpr - fpr)]  # Youden's J statistic
    y_pred = (y_prob > best_thresh).astype(int)

    auc = roc_auc_score(y_val, y_prob)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    report["AUC"] = auc
    reports.append(report)

    print(f"Best threshold for fold {fold}: {best_thresh:.2f}, AUC={auc:.4f}")
    K.clear_session()

# 平均结果
avg_report = {}
for key in reports[0].keys():
    if isinstance(reports[0][key], dict):
        avg_report[key] = {m: np.mean([r[key][m] for r in reports]) for m in reports[0][key]}
    else:
        avg_report[key] = np.mean([r[key] for r in reports])

print("\n=== Average Classification Report ===")
rows = {}
for k, v in avg_report.items():
    if isinstance(v, dict):
        rows[k] = [v['precision'], v['recall'], v['f1-score']]
    else:
        rows[k] = [v, v, v]

df_report = pd.DataFrame.from_dict(
    rows, orient="index", columns=["precision", "recall", "f1-score"]
)
print(df_report.to_string(float_format="%.6f"))

results.append({
    "params": params,
    "Accuracy": avg_report["accuracy"],
    "F1_macro": avg_report["macro avg"]["f1-score"],
    "Recall_0": avg_report["0"]["recall"],
    "Recall_1": avg_report["1"]["recall"],
    "AUC": avg_report["AUC"]
})

print("\n*** Final Results ***")
print(pd.DataFrame(results))
