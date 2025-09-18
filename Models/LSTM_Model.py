import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Read the Hawaii dataset
df = pd.read_csv("normalized_Hawaii_Training_Data_Cleaned.csv")
y = df["outage_flag"].values
X = df.drop(columns=["outage_flag"]).values

# reshape → number of samples, timesteps=24, number of features=6
n_features = 6
n_steps = 24
X = X.reshape(-1, n_steps, n_features)

# 10-fold CV
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []

fold = 1
for train_idx, val_idx in kf.split(X, y):
    print(f"\nFold {fold}")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Build a new LSTM model for each fold
    model = Sequential()
    model.add(LSTM(64, input_shape=(n_steps, n_features), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Training the model
    model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # Prediction & Evaluation
    y_pred = (model.predict(X_val) > 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    acc_scores.append(acc)
    prec_scores.append(prec)
    rec_scores.append(rec)
    f1_scores.append(f1)

    K.clear_session()
    fold += 1

# Output average results
print("\n** LSTM 10-Fold CV Average Results **")
print(f"Accuracy: {np.mean(acc_scores):.4f}")
print(f"Precision: {np.mean(prec_scores):.4f}")
print(f"Recall: {np.mean(rec_scores):.4f}")
print(f"F1-score: {np.mean(f1_scores):.4f}")
