from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K



class LSTMModel:
    def __init__(self, params):
        self.params = params
        self.model = self.build_model(params)

    def build_model(self, params):

        model = Sequential([
            Input(shape=(24, 6)),
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
    
    def fit(self, X, y):
        self.model.fit(
            X, y,
            epochs=15,
            batch_size=params["batch_size"],
            validation_split=0.2,
            verbose=0
        )
    
    def reshapeInput(self, X):
        return X.reshape((X.shape[0], 24, 6))
    
    def predict(self, X):
        X = self.reshapeInput(X)
        return (self.model.predict(X, verbose=0) > 0.5).astype(int)