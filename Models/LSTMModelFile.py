from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd



class LSTMEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, lstm_units=64, dropout=0.5, lr=5e-4, batch_size=32, epochs=15):
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(24, 6)),   # ⚠️ assumes fixed input shape
            LSTM(self.lstm_units, return_sequences=True),
            Dropout(self.dropout),
            LSTM(self.lstm_units // 2),
            Dropout(self.dropout),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model
    
    def fit(self, X, y):
        X = self.reshapeInput(X)
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )

    def predict_proba(self, X):
        X = self.reshapeInput(X)
        probs = self.model.predict(X, verbose=0)
        return np.hstack([1 - probs, probs])
    
    def reshapeInput(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return X.reshape((X.shape[0], 24, 6))
    
    def predict(self, X):
        X = self.reshapeInput(X)
        return (self.model.predict(X, verbose=0) > 0.5).astype(int)