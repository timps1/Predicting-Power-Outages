import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os, warnings

# 基本设置
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# 可调参数

param = {
    "lstm_units": 64,
    "dropout": 0.3,
    "lr": 1e-3,
    "batch_size": 32,
    "epochs": 15,
}

sus_majority_to_minority_ratio = 2.0
sus_k = 5
n_steps = 24   # 每个序列的时间步

# 读取与预处理数据
df = pd.read_csv("ALL_USA_TRAINING_DATA.csv")
y = df["outage_flag"].values
X = df.drop(columns=["outage_flag", "date"], errors="ignore")

# 转数值 + 填充 NaN
X = X.apply(pd.to_numeric, errors="coerce").fillna(0).values

# 变形为 3D: [样本数, 时间步, 每步特征]
n_features = X.shape[1] // n_steps
X = X.reshape(-1, n_steps, n_features)

print(f"[INFO] Data shape: X={X.shape}, y={y.shape}")
print(f"[INFO] Outage distribution: {np.bincount(y)}")

# 自定义选择性欠采样 (SUS)
def selective_undersample_by_distance(X_seq, y_vec, majority_to_minority_ratio=2.0, k=5):
    X_flat = X_seq.reshape((X_seq.shape[0], -1))
    classes, counts = np.unique(y_vec, return_counts=True)
    maj_label = classes[np.argmax(counts)]
    min_label = classes[np.argmin(counts)]

    idx_maj = np.where(y_vec == maj_label)[0]
    idx_min = np.where(y_vec == min_label)[0]

    X_min = X_flat[idx_min]
    X_maj = X_flat[idx_maj]

    nn = NearestNeighbors(n_neighbors=min(k, len(idx_min)), algorithm="auto")
    nn.fit(X_min)
    dist, _ = nn.kneighbors(X_maj)
    hardness = dist.min(axis=1)  # 越小越难
    n_min = len(idx_min)
    n_maj_keep = int(np.clip(majority_to_minority_ratio * n_min, 1, len(idx_maj)))
    top_idx = np.argsort(hardness)[:n_maj_keep]
    keep_majority_idx = idx_maj[top_idx]

    selected = np.concatenate([idx_min, keep_majority_idx])
    np.random.shuffle(selected)
    return X_seq[selected], y_vec[selected]

# 采样（选择性欠采样）
X_train, y_train = selective_undersample_by_distance(
    X, y,
    majority_to_minority_ratio=sus_majority_to_minority_ratio,
    k=sus_k
)
print("[INFO] After SUS:", np.bincount(y_train))

# 构建 LSTM 模型
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

model = build_model()
model.compile(
    optimizer=Adam(learning_rate=param["lr"]),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# 训练
history = model.fit(
    X_train, y_train,
    epochs=param["epochs"],
    batch_size=param["batch_size"],
    validation_split=0.1,
    verbose=1,
    class_weight={0: 1, 1: 30}  # 少数类权重补偿
)

# 模型评估
y_prob = model.predict(X_train).ravel()
fpr, tpr, thresholds = roc_curve(y_train, y_prob)
best_thresh = thresholds[np.argmax(tpr - fpr)]
y_pred = (y_prob > best_thresh).astype(int)

auc = roc_auc_score(y_train, y_prob)
report = classification_report(y_train, y_pred, digits=4)
print("\n================= Training Report =================")
print(report)
print(f"AUC = {auc:.4f}")
print(f"Best threshold = {best_thresh:.3f}")
print("===================================================")


# 导出模型
model.save("usa_lstm_model.h5")
print("\nUSA LSTM model exported successfully → usa_lstm_model.h5")

with open("usa_model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))
    f.write("\n\nClassification Report:\n")
    f.write(report)
    f.write(f"\nAUC: {auc:.4f}\nThreshold: {best_thresh:.3f}\n")

print("Model summary and metrics saved to usa_model_summary.txt")
