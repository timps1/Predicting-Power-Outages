import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from Models.SklearnModels import sklearnModel  # keep same parent interface


class LSTMModel(sklearnModel):
    """
    Bidirectional LSTM model wrapper for ensemble integration.
    Supports training, prediction, saving, and loading.
    """
    def __init__(
        self,
        n_steps=24,
        lstm_units=64,
        dropout=0.3,
        lr=1e-3,
        batch_size=32,
        epochs=15,
        class_weight={0: 1, 1: 30},
        model_path="usa_lstm_model.h5"
    ):
        self.n_steps = n_steps
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_weight = class_weight
        self.model_path = model_path
        self.model = None

    # Core model builder
    def build_model(self, n_features):
        model = Sequential([
            Input(shape=(self.n_steps, n_features)),
            Bidirectional(LSTM(self.lstm_units, return_sequences=True)),
            Dropout(self.dropout),
            Bidirectional(LSTM(self.lstm_units // 2)),
            Dropout(self.dropout),
            Dense(1, activation="sigmoid")
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # Training
    def train(self, X, y):
        """
        X: numpy array [samples, timesteps, features]
        y: numpy array [samples]
        """
        n_features = X.shape[2]
        self.model = self.build_model(n_features)

        es = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        history = self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=1,
            class_weight=self.class_weight,
            callbacks=[es]
        )
        return history

    # Prediction (consistent with sklearn predict_proba)
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not loaded or trained yet.")
        y_prob = self.model.predict(X, verbose=0).ravel()
        return np.vstack([1 - y_prob, y_prob]).T  # shape [N,2]

    def predict(self, X, threshold=0.5):
        y_prob = self.predict_proba(X)[:, 1]
        return (y_prob > threshold).astype(int)

    # Evaluation
    def evaluate(self, X, y):
        y_prob = self.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        y_pred = (y_prob > best_thresh).astype(int)

        auc = roc_auc_score(y, y_prob)
        report = classification_report(y, y_pred, digits=4)
        print("\n=== Evaluation Report ===")
        print(report)
        print(f"AUC = {auc:.4f}")
        print(f"Best threshold = {best_thresh:.3f}")
        return {"AUC": auc, "threshold": best_thresh, "report": report}

    # Save / Load
    def save(self, path=None):
        if path is None:
            path = self.model_path
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path=None):
        if path is None:
            path = self.model_path
        self.model = load_model(path)
        print(f"Model loaded from {path}")

    # Type tag for ensemble identification
    def getType(self):
        return "LSTM"
