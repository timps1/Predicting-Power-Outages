import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# ================================
# 1. Load dataset
# ================================
df = pd.read_csv("NZ_all_normalised_cleaned.csv")

# Drop non-feature columns
X = df.drop(columns=["outage_flag", "Date", "date_only", "region"]).values
y = df["outage_flag"].values

# Reshape → (samples, time steps, features)
n_steps = 24
n_features = 6
X = X.reshape(-1, n_steps, n_features)

print("X shape:", X.shape, "y shape:", y.shape)
print("Outage distribution:", np.bincount(y))

# ================================
# 2. Define function to build model
# ================================
def build_lstm(params):
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(params["lstm_units"], return_sequences=True),
        Dropout(params["dropout"]),
        LSTM(params["lstm_units"] // 2),
        Dropout(params["dropout"]),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=params["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ================================
# 3. Hyperparameters
# ================================
param_grid = [
    {"lstm_units": 32, "dropout": 0.3, "lr": 1e-3, "batch_size": 32},
    # {"lstm_units": 64, "dropout": 0.3, "lr": 1e-3, "batch_size": 32},
]

# ================================
# 4. Training with StratifiedKFold
# ================================
results = []

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for params in param_grid:
    print(f"\n=== Evaluating LSTM with params: {params} ===")
    reports = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n--- Fold {fold} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 方式一: 使用 class_weight
        class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y_train)
        class_weights = dict(zip(np.unique(y), class_weights))

        model = build_lstm(params)

        model.fit(
            X_train, y_train,
            epochs=15,
            batch_size=params["batch_size"],
            validation_split=0.2,
            verbose=0,
            class_weight=class_weights
        )

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

    print("\nAverage Classification Report:")
    print(pd.DataFrame(avg_report).T)

    results.append({
        "params": params,
        "Accuracy": avg_report["accuracy"],
        "F1_macro": avg_report["macro avg"]["f1-score"],
        "Recall_1": avg_report["1"]["recall"]
    })

# ================================
# 5. Final results
# ================================
results_df = pd.DataFrame(results)
print("\n*** Summary Results ***")
print(results_df)

best_idx = results_df["F1_macro"].idxmax()
print("\nBest config:", results_df.loc[best_idx])
