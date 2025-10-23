import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE   # 重点：平衡数据
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import warnings, os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# ========== 1. 读取数据 ==========
df = pd.read_csv("NZ_all_normalised_cleaned.csv")
y = df["outage_flag"].values
X = df.drop(columns=["outage_flag", "Date", "date_only", "region"]).values  # 只保留数值特征
# 转 DataFrame 方便处理缺失
X = pd.DataFrame(X).astype(float)

# 方法 1：简单用列均值填充
X = X.fillna(X.mean())

# 转回 numpy
X = X.values

print("原始数据:", X.shape, y.shape)
print("Outage 分布:", np.bincount(y))

# ========== 2. SMOTE 过采样 ==========
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("SMOTE 后数据:", X_res.shape, y_res.shape)
print("Outage 分布:", np.bincount(y_res))

# reshape → (samples, time steps, features)
n_steps = 24
n_features = 6
X_res = X_res.reshape(-1, n_steps, n_features)

# ========== 3. 定义模型 ==========
def make_model(lstm_units, dropout, lr):
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout),
        LSTM(lstm_units // 2),
        Dropout(dropout),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# ========== 4. K-fold 训练 ==========
param_grid = [
    {"lstm_units": 32, "dropout": 0.3, "lr": 1e-3, "batch_size": 32}
]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
reports = []

for params in param_grid:
    print(f"\n=== Evaluating LSTM with params: {params} ===")
    fold = 1
    for train_idx, val_idx in kf.split(X_res, y_res):
        print(f"--- Fold {fold} ---")
        fold += 1
        X_train, X_val = X_res[train_idx], X_res[val_idx]
        y_train, y_val = y_res[train_idx], y_res[val_idx]

        model = make_model(params["lstm_units"], params["dropout"], params["lr"])
        model.fit(X_train, y_train, epochs=10,
                  batch_size=params["batch_size"], verbose=0)

        y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        reports.append(report)
        K.clear_session()

    # 平均结果
    avg_report = {}
    for key in reports[0].keys():
        if isinstance(reports[0][key], dict):
            avg_report[key] = {m: np.mean([r[key][m] for r in reports]) for m in reports[0][key]}
        else:
            avg_report[key] = np.mean([r[key] for r in reports])

    # 打印平均分类报告
    print("=== Average Classification Report ===")

    rows = {}
    for k, v in avg_report.items():
        if isinstance(v, dict):  # precision, recall, f1-score 部分
            rows[k] = [v['precision'], v['recall'], v['f1-score']]
        elif k == "accuracy":  # 特殊处理 accuracy
            rows[k] = [v, v, v]

    df_report = pd.DataFrame.from_dict(
        rows, orient="index", columns=["precision", "recall", "f1-score"]
    )

    print(df_report.to_string(float_format="%.6f"))

