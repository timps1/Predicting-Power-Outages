import sklearn
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from Models.SklearnModels import sklearnModel
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif




# from Data.Data_Preprocessing.Data_Cleaning import extract_number

import re

def extract_number(text, pattern=r"(-?\d+(?:\.\d+)?)"):
    """Extract the first number using regex; return NA if not found."""
    if pd.isna(text):
        return pd.NA
    m = re.search(pattern, str(text))
    return float(m.group(1)) if m else pd.NA









df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../Data/normalized_Hawaii_Training_Data_Cleaned.csv"))

# for i in range(24):
#     # df = df.drop([f"Time_{i}"], axis=1)
#     df[f"Humidity_{i}"] = pd.to_numeric(df[f"Humidity_{i}"].apply(lambda x: extract_number(x)/100), errors="coerce")
#     df[f"Barometer_{i}"] = pd.to_numeric(df[f"Barometer_{i}"].apply(lambda x: extract_number(x)), errors="coerce")
    

X = df.drop(["outage_flag"], axis=1)
Y = df["outage_flag"]

# sel = VarianceThreshold(.5*(1-.5)) #50% variance
# X = sel.fit_transform(X)

# X = SelectKBest(f_classif, k=6).fit_transform(X_pre, Y)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


xgbClf = xgb.XGBClassifier(gamma = 0, max_depth = 20)


param_grid = {'max_depth': [3, 18, 25, 50, 100],
              'learning_rate': [0.1, 0.01, 0.3, 0.5],
              'gamma': [0, 1, 5, 10],
            #   'booster': ['gbtree', 'gblinear', 'dart'],
              'subsample': [0, 0.2, 0.5, 1]}

grid_search = GridSearchCV(xgbClf, param_grid, cv=10, scoring='accuracy')
grid_search.fit(x_train, y_train)

print("Best set of hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# xgbClf.fit(x_train, y_train)

# y_pred = xgbClf.predict(x_test)

# print(classification_report(y_test, y_pred))

# print(f"No CV accuracy:", accuracy_score(y_pred, y_test))

#CV
# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# cv_acc = cross_val_score(xgbClf, X, Y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
# print(f"10-fold CV accuracy:", cv_acc)
# cv_nrmse = cross_val_score(xgbClf, X, Y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1).mean()
# print(f"10-fold CV -RMSE:", cv_nrmse)
# cv_f1 = cross_val_score(xgbClf, X, Y, cv=cv, scoring="f1", n_jobs=-1).mean()
# print(f"10-fold CV f1:", cv_f1)




