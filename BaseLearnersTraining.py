import sys
from Models.SklearnModels import sklearnModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
from xgboost import XGBClassifier
from Models.SVMtorch import SVM
import time
from CrossValidationMethod import MethodOfCrossValidation
from sklearn.ensemble import ExtraTreesClassifier
from Data_Preprocessing.SUS import applySUS
from Data_Preprocessing.smote_method import apply_smote
import joblib
import InputLineManagement as ilm

def buildAndSaveBaseLearners(X, y):

    
    rf_params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1
    }

    svm_params = {
        "n_features" : 4000, 
        "gamma" : "scale", 
        "lr" : 0.01, 
        "weight_decay" : 0.005
    }

    knn_params = {
        "n_neighbors": 2,
        "weights": "distance",
        "metric": "manhattan"
    }

    xgb_params = {
        "learning_rate": 0.3,
        "gamma": 0,
        "max_depth": 6,
        "n_estimators": 100,
        "subsample": 1,
        "random_state": 42,
        "n_jobs": -1
    }

    et_params = {
        "n_estimators" : 5, 
        "criterion" : 'entropy', 
        "max_features" : 2
    }

    modelClassParameterList = [
        (RandomForestClassifier(**rf_params), "rf"), 
        (KNeighborsClassifier(**knn_params), "knn"),
        (XGBClassifier(**xgb_params), "xgb"),
        # (SVM(**svm_params), "svm"),
        (ExtraTreesClassifier(**et_params), "et"),
        ]

    X, y = applySUS(X, y)
    X, y = apply_smote(X, y)

    additionalLabel = ilm.getArg("MODEL-SAVE-LABEL")
    fileInfo = ""

    for i, (model, tag) in enumerate(modelClassParameterList):

        print(f"------------------------------------------------------------")
        print(f"Model {i}: {model.get_params()}")

        model.fit(X,y)

        if tag == "svm":
            model_filename = f"Trained_Models/{tag}_model.pth"
            model.save(model_filename)
        else:
            model_filename = f"Trained_Models/{tag}_model.joblib"
            joblib.dump(model, model_filename)
            print("Model saved!", model_filename)
        
        fileInfo += f'{model_filename},{tag}\n'

        print(f"------------------------------------------------------------")
    
    if ilm.getArg("MODELS-FILENAME-WRITE") is not None and len(fileInfo) > 0:
        aFile = open(ilm.getArg("MODELS-FILENAME-WRITE"), "w")
        aFile.write(fileInfo)
        aFile.close()

    
def trainBaseLearners(X, y):

    rf_params = {
        "n_estimators": 200,
        "random_state": 42,
        "n_jobs": -1
    }

    svm_params = {
        "n_features" : 700, 
        "gamma" : "scale", 
        "lr" : 0.01, 
        "weight_decay" : 0.1/40
    }

    knn_params = {
        "n_neighbors": 2,
        "weights": "distance",
        "metric": "manhattan"
    }

    xgb_params = {
        "learning_rate": 0.3,
        "gamma": 0,
        "max_depth": 6,
        "n_estimators": 100,
        "subsample": 1,
        "random_state": 42,
        "n_jobs": -1
    }

    et_params = {
        "n_estimators" : 5, 
        "criterion" : 'entropy', 
        "max_features" : 2
    }

    modelClassParameterList = [
        (RandomForestClassifier(**rf_params), "rf"), 
        (KNeighborsClassifier(**knn_params), "knn"),
        (XGBClassifier(**xgb_params), "xgb"),
        # (SVM(**svm_params), "svm"),
        (ExtraTreesClassifier(**et_params), "et"),
        ]
    
    MethodOfCrossValidation(X, y, modelClassParameterList)
