# ==========================
# 1. Import libraries
# ==========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ==========================
# 2. Load dataset
# ==========================
df = pd.read_csv("data/da/normalized_Hawaii_Training_Data_Cleaned.csv")

# Define features and target
X = df.drop(columns=["outage_flag"])
y = df["outage_flag"]

# ==========================
# 3. Train-test split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================
# 4. Apply SMOTE
# ==========================
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

print("Original training distribution:\n", y_train.value_counts())
print("\nAfter SMOTE:\n", y_res.value_counts())

# ==========================
# 5. Model evaluation helper
# ==========================
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n===== {model_name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# ==========================
# 6. Train & Evaluate Models
# ==========================

# 6.1 K-NN
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn, X_res, y_res, X_test, y_test, "K-NN")

# 6.2 SVM Linear
svm_linear = SVC(kernel="linear", random_state=42)
evaluate_model(svm_linear, X_res, y_res, X_test, y_test, "SVM (Linear)")

# 6.3 SVM RBF
svm_rbf = SVC(kernel="rbf", random_state=42)
evaluate_model(svm_rbf, X_res, y_res, X_test, y_test, "SVM (RBF)")

# 6.4 Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
evaluate_model(gb, X_res, y_res, X_test, y_test, "Gradient Boosting")

# 6.5 XGBoost
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
evaluate_model(xgb_clf, X_res, y_res, X_test, y_test, "XGBoost")

# 6.6 Random Forest
rf = RandomForestClassifier(random_state=42)
evaluate_model(rf, X_res, y_res, X_test, y_test, "Random Forest")

# ==========================
# 7. LSTM (需要深度学习)
# ==========================
# 注意：LSTM 需要三维输入 [samples, timesteps, features]
# 这里简化为每个样本一个时间步
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)
X_test_scaled = scaler.transform(X_test)

# reshape for LSTM
X_res_lstm = X_res_scaled.reshape((X_res_scaled.shape[0], 1, X_res_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

y_res_cat = to_categorical(y_res)
y_test_cat = to_categorical(y_test)

from tensorflow.keras.layers import LSTM

model_lstm = Sequential()
model_lstm.add(LSTM(64, input_shape=(1, X_res.shape[1]), return_sequences=False))
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(32, activation="relu"))
model_lstm.add(Dense(y_res_cat.shape[1], activation="softmax"))

model_lstm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("\n===== LSTM Training =====")
model_lstm.fit(X_res_lstm, y_res_cat, epochs=10, batch_size=32, verbose=1)

loss, acc = model_lstm.evaluate(X_test_lstm, y_test_cat, verbose=0)
print("\n===== LSTM =====")
print("Accuracy:", acc)
