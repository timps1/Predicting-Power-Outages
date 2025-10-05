import sys
from Models.SklearnModels import sklearnModel
from IO_Data import IOData
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from Model_Evaluation import CrossValidation
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
# from Models.LSTMModelFile import LSTMEstimator
import pandas as pd
from xgboost import XGBClassifier
from Models.SVMtorch import SVM
import joblib
from CrossValidationMethod import MethodOfCrossValidation


def trainingEnsemble(X, y):

    finalEstimatorTag = "rf"
    passthrough = True

    rf_meta_params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1
    }

    xgboost_meta_params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1
    }

    modelFilenamesAndTags = [
        ("Trained_Models/rf_model.joblib", "rf"),
        ("Trained_Models/svm_model.pth", "svm"),
        ("Trained_Models/xgb_model.joblib", "xgb"),
        ("Trained_Models/knn_model.joblib", "knn")
    ]

    loadedModels = []

    for i, (modelFilename, tag) in enumerate(modelFilenamesAndTags):
        if tag == "svm":
            model = SVM()
            model.load(modelFilename)
        else: 
            model = joblib.load(modelFilename)
        loadedModels.append(model)
        print(f"Model {tag} loaded!")
    

    predictionsList = []
    for i, model in enumerate(loadedModels):
        predictionsList.append(model.predict_proba(X)[:, 1])
    
    prediction_cols = [
        pd.Series(preds, name=f"{modelFilenamesAndTags[i][1]} prediction")
        for i, preds in enumerate(predictionsList)
    ]

    if passthrough:
        X = pd.concat([X] + prediction_cols, axis=1)
    else:
        X = pd.concat(prediction_cols, axis=1)

    meta_model = RandomForestClassifier(**rf_meta_params)

    MethodOfCrossValidation(X, y, [(meta_model, finalEstimatorTag)])
    

def fitAndSave(X, y, meta_model, finalEstimatorTag):    
    meta_model.fit(X, y)

    model_filename = f"{finalEstimatorTag}_ensemble_model.joblib"
    joblib.dump(meta_model, model_filename)
    print("Model saved!", model_filename)

