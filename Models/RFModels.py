import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


file_path = "normalized_Hawaii_Training_Data_Cleaned.csv"
df = pd.read_csv(file_path)


X = df.drop(columns=["outage_flag"])
y = df["outage_flag"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 训练随机森林
rf = RandomForestClassifier(
    n_estimators=200,     # 树的数量
    max_depth=None,      # 树深度，不限制
    random_state=42,
    n_jobs=-1            # 并行训练
)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Outage", "Outage"], yticklabels=["No Outage", "Outage"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()


importances = rf.feature_importances_
indices = importances.argsort()[::-1][:20]
plt.figure(figsize=(10,6))
plt.bar(range(20), importances[indices], align="center")
plt.xticks(range(20), [X.columns[i] for i in indices], rotation=90)
plt.title("Top 20 Feature Importances (Random Forest)")
plt.show()

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Features and target
X = df.drop(columns=["outage_flag"])
y = df["outage_flag"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Run RF with different number of trees to mimic KNN varying k
results = []
for n_estimators in [50, 100, 200, 300, 400, 500]:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    # Train on training split
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Holdout metrics
    acc_holdout = accuracy_score(y_test, y_pred)
    f1_holdout = f1_score(y_test, y_pred)

    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    acc_cv = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
    f1_cv = cross_val_score(rf, X, y, cv=cv, scoring="f1", n_jobs=-1).mean()

    results.append((n_estimators, acc_holdout, f1_holdout, acc_cv, f1_cv))

    print(f"n_estimators={n_estimators:3d}, "
          f"Holdout: ACC={acc_holdout:.4f}, F1={f1_holdout:.4f} | "
          f"10-CV: ACC={acc_cv:.4f}, F1={f1_cv:.4f}")