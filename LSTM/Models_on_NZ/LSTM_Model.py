import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import warnings
import os         # to hide annoying logs and warnings...

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Read the Hawaii dataset
df = pd.read_csv("normalized_NZ_Training_Data_Cleaned.csv")
y = df["outage_flag"].values
X = df.drop(columns=["outage_flag"]).values

# reshape → number of samples, time steps = 24, number of features = 6
n_features = 6
n_steps = 24
X = X.reshape(-1, n_steps, n_features)

# Hyperparameters
param_grid = [
    {"lstm_units": 32, "dropout": 0.3, "lr": 1e-3, "batch_size": 32},
    {"lstm_units": 64, "dropout": 0.3, "lr": 1e-3, "batch_size": 32},
    {"lstm_units": 64, "dropout": 0.5, "lr": 5e-4, "batch_size": 32},
    {"lstm_units": 128, "dropout": 0.5, "lr": 5e-4, "batch_size": 64},
]

results = []

# Iterate over hyperparameter combinations
for params in param_grid:
    print(f"*** Evaluating LSTM with params: {params} ***")

    reports = []

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # LSTM
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

        # train
        model.fit(
            X_train, y_train,
            epochs=15,
            batch_size=params["batch_size"],
            validation_split=0.2,
            verbose=0
        )

        # predict
        y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int)

        # save
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        reports.append(report)
        K.clear_session()

    #  Averaging
    avg_report = {}
    for key in reports[0].keys():
        if isinstance(reports[0][key], dict):
            avg_report[key] = {metric: np.mean([r[key][metric] for r in reports]) for metric in reports[0][key]}
        else:
            avg_report[key] = np.mean([r[key] for r in reports])

    # print (for 1 hyperparameter combination)
    print(f"Average Score across 10 folds: {avg_report['accuracy']:.6f}")
    rows = {}
    for k, v in avg_report.items():
        if isinstance(v, dict):
            rows[k] = [v['precision'], v['recall'], v['f1-score']]
        else:  # accuracy
            rows[k] = [v, v, v]

    df_report = pd.DataFrame.from_dict(
        rows, orient="index", columns=["precision", "recall", "f1-score"]
    )
    print("Average Classification Report:")
    print(df_report.to_string(float_format="%.6f"))

    results.append({
        "params": params,
        "Accuracy": avg_report["accuracy"],
        "Precision_macro": avg_report["macro avg"]["precision"],
        "Recall_macro": avg_report["macro avg"]["recall"],
        "F1_macro": avg_report["macro avg"]["f1-score"]
    })

# Summary table
results_df = pd.DataFrame(results)
print("\n*** Final Hyperparameter Search Results ***")
print(results_df)

# Optimal choice
best_idx = results_df["F1_macro"].idxmax()
best_params = results_df.loc[best_idx]
print("\nBest LSTM configuration:")
print(best_params)
