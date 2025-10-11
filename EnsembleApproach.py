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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import InputLineManagement as ilm


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

    inputModelsFile = ilm.getArg("MODELS-FILENAME-READ", "r")
    modelsWithTags = inputModelsFile.read().strip().split("\n")
    inputModelsFile.close()

    modelFilenamesAndTags = [tuple(modelAndTag.strip().split(",")) for modelAndTag in modelsWithTags]

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

    fitAndSave(X, y, meta_model, finalEstimatorTag)
    

def fitAndSave(X, y, meta_model, finalEstimatorTag):    
    meta_model.fit(X, y)

    model_filename = f"Trained_Models/{finalEstimatorTag}_ensemble_model.joblib"
    joblib.dump(meta_model, model_filename)
    print("Model saved!", model_filename)

def openAndPredict(X, y, filename = None):

    passthrough = True

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
    
    if filename is None:
        filename = "Trained_Models/rf_ensemble_model.joblib"

    model = joblib.load(filename)

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

