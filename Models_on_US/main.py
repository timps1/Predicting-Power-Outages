from Models_on_US.US_LSTM_ensemble import LSTMModel

# Load data (already shaped to [N, 24, n_features])
lstm = LSTMModel()

# Train
lstm.train(X_train, y_train)

# Evaluate
lstm.evaluate(X_val, y_val)

# Save predictions for ensemble
pred_probs = lstm.predict_proba(X_test)[:, 1]
pd.DataFrame({"id": ids, "lstm_prob": pred_probs}).to_csv("lstm_predictions.csv", index=False)

# Save model
lstm.save()
