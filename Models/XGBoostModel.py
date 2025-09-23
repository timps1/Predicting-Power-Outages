import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split,  GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif





import re

def extract_number(text, pattern=r"(-?\d+(?:\.\d+)?)"):
    """Extract the first number using regex; return NA if not found."""
    if pd.isna(text):
        return pd.NA
    m = re.search(pattern, str(text))
    return float(m.group(1)) if m else pd.NA









df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../Data/normalized_Hawaii_Training_Data_Cleaned.csv"))
    

X = df.drop(["outage_flag"], axis=1)
Y = df["outage_flag"]


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


xgbClf = xgb.XGBClassifier()

# # Hyperparameter Tuning
# param_grid = {'max_depth': [3, 18, 25, 50, 100],
#               'learning_rate': [0.1, 0.3, 0.5, 0.75, 1],
#               'gamma': [0, 1, 5, 10],
#             #   'booster': ['gbtree', 'gblinear', 'dart'],
#               'subsample': [0, 0.2, 0.5, 1]}

# grid_search = GridSearchCV(xgbClf, param_grid, cv=10, scoring='accuracy')
# grid_search.fit(x_train, y_train)

# print("Best set of hyperparameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)

# xgbClf.set_params(**grid_search.best_params_)




num_estimators = np.arange(1,100, 5)



accuracies = []
precisions = []
recalls = []
f1s = []


for num_estimator in num_estimators:
  xgbClf.set_params(n_estimators = num_estimator)
  xgbClf.fit(x_train, y_train)
  y_pred = xgbClf.predict(x_test)


  # CV
  cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  cv_acc = cross_val_score(xgbClf, X, Y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
  cv_pre = cross_val_score(xgbClf, X, Y, cv=cv, scoring="precision", n_jobs=-1).mean()
  cv_rec = cross_val_score(xgbClf, X, Y, cv=cv, scoring="recall", n_jobs=-1).mean()
  # cv_nrmse = cross_val_score(xgbClf, X, Y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1).mean()
  cv_f1 = cross_val_score(xgbClf, X, Y, cv=cv, scoring="f1", n_jobs=-1).mean()

  accuracies.append(cv_acc)
  precisions.append(cv_pre)
  recalls.append(cv_rec)
  f1s.append(cv_f1)


plt.plot(num_estimators, np.array(accuracies)*100, "-o", label="Accuracy")
plt.plot(num_estimators, np.array(precisions)*100, "-o",label="Precision")
plt.plot(num_estimators, np.array(recalls)*100, "-o",label="Recall")
plt.plot(num_estimators, np.array(f1s)*100, "-o",label="F1 Score")
plt.xlabel("n_estimators")
plt.ylabel("%")
plt.title("XGBoost Classifier Performance vs n_estimators")
plt.legend()
plt.grid()
plt.ylim(60,100)
plt.show()





max_depths = np.arange(1,105, 10)



accuracies = []
precisions = []
recalls = []
f1s = []


for max_depth in max_depths:
  xgbClf.set_params(max_depth=max_depth)
  xgbClf.fit(x_train, y_train)
  y_pred = xgbClf.predict(x_test)


  # CV
  cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  cv_acc = cross_val_score(xgbClf, X, Y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
  cv_pre = cross_val_score(xgbClf, X, Y, cv=cv, scoring="precision", n_jobs=-1).mean()
  cv_rec = cross_val_score(xgbClf, X, Y, cv=cv, scoring="recall", n_jobs=-1).mean()
  # cv_nrmse = cross_val_score(xgbClf, X, Y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1).mean()
  cv_f1 = cross_val_score(xgbClf, X, Y, cv=cv, scoring="f1", n_jobs=-1).mean()

  accuracies.append(cv_acc)
  precisions.append(cv_pre)
  recalls.append(cv_rec)
  f1s.append(cv_f1)


plt.plot(max_depths, np.array(accuracies)*100, "-o", label="Accuracy")
plt.plot(max_depths, np.array(precisions)*100, "-o",label="Precision")
plt.plot(max_depths, np.array(recalls)*100, "-o",label="Recall")
plt.plot(max_depths, np.array(f1s)*100, "-o",label="F1 Score")
plt.xlabel("max_depth")
plt.ylabel("%")
plt.title("XGBoost Classifier Performance vs max_depth")
plt.legend()
plt.grid()
plt.ylim(60,100)
plt.show()





learning_rates = np.arange(0.1, 1.1, 0.1)

accuracies = []
precisions = []
recalls = []
f1s = []


for learning_rate in learning_rates:
  xgbClf.set_params(max_depth=20, learning_rate=learning_rate)
  xgbClf.fit(x_train, y_train)
  y_pred = xgbClf.predict(x_test)


  # CV
  cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  cv_acc = cross_val_score(xgbClf, X, Y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
  cv_pre = cross_val_score(xgbClf, X, Y, cv=cv, scoring="precision", n_jobs=-1).mean()
  cv_rec = cross_val_score(xgbClf, X, Y, cv=cv, scoring="recall", n_jobs=-1).mean()
  # cv_nrmse = cross_val_score(xgbClf, X, Y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1).mean()
  cv_f1 = cross_val_score(xgbClf, X, Y, cv=cv, scoring="f1", n_jobs=-1).mean()

  accuracies.append(cv_acc)
  precisions.append(cv_pre)
  recalls.append(cv_rec)
  f1s.append(cv_f1)


plt.plot(learning_rates, np.array(accuracies)*100, "-o", label="Accuracy")
plt.plot(learning_rates, np.array(precisions)*100, "-o",label="Precision")
plt.plot(learning_rates, np.array(recalls)*100, "-o",label="Recall")
plt.plot(learning_rates, np.array(f1s)*100, "-o",label="F1 Score")
plt.xlabel("Learning Rate")
plt.ylabel("%")
plt.title("XGBoost Classifier Performance vs Learning Rate")
plt.legend()
plt.grid()
plt.ylim(60,100)
plt.show()


gammas = np.arange(0, 15, 1)

accuracies = []
precisions = []
recalls = []
f1s = []


for gamma in gammas:
  xgbClf.set_params(learning_rate=0.1, gamma=gamma)
  xgbClf.fit(x_train, y_train)
  y_pred = xgbClf.predict(x_test)


  # CV
  cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  cv_acc = cross_val_score(xgbClf, X, Y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
  cv_pre = cross_val_score(xgbClf, X, Y, cv=cv, scoring="precision", n_jobs=-1).mean()
  cv_rec = cross_val_score(xgbClf, X, Y, cv=cv, scoring="recall", n_jobs=-1).mean()
  # cv_nrmse = cross_val_score(xgbClf, X, Y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1).mean()
  cv_f1 = cross_val_score(xgbClf, X, Y, cv=cv, scoring="f1", n_jobs=-1).mean()

  accuracies.append(cv_acc)
  precisions.append(cv_pre)
  recalls.append(cv_rec)
  f1s.append(cv_f1)


plt.plot(gammas, np.array(accuracies)*100, "-o", label="Accuracy")
plt.plot(gammas, np.array(precisions)*100, "-o",label="Precision")
plt.plot(gammas, np.array(recalls)*100, "-o",label="Recall")
plt.plot(gammas, np.array(f1s)*100, "-o",label="F1 Score")
plt.xlabel("gamma")
plt.ylabel("%")
plt.title("XGBoost Classifier Performance vs gamma")
plt.legend()
plt.grid()
plt.ylim(60,100)
plt.show()


