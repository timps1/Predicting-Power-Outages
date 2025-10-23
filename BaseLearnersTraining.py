from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from xgboost import XGBClassifier
from CrossValidationMethod import MethodOfCrossValidation
from sklearn.ensemble import ExtraTreesClassifier
from Data_Preprocessing.SUS import applySUS
from Data_Preprocessing.smote_method import apply_smote
import joblib

def buildAndSaveBaseLearners(X, y):
    """
    This function builds and saves multiple base learner models.
    It defines a list of models with their parameters, trains each model
    on the provided dataset (X, y), and saves the trained models to Trained_Models folder.
    Args:
        X (pd.DataFrame): Feature dataset.
        y (pd.Series): Target labels.
    Returns:
        None
    """
    
    rf_params = {
        "n_estimators": 200,
        "random_state": 42,
        "n_jobs": -1
    }

    knn_params = {
        "n_neighbors": 4,
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
        "n_estimators" : 80, 
        "criterion" : 'entropy', 
        "max_features" : 80,
        "random_state" : 42
    }

    modelClassParameterList = [
        (RandomForestClassifier(**rf_params), "rf"), 
        (KNeighborsClassifier(**knn_params), "knn"),
        (XGBClassifier(**xgb_params), "xgb"),
        (ExtraTreesClassifier(**et_params), "et"),
        ]

    X, y = applySUS(X, y, 0.2)
    X, y = apply_smote(X, y, 1)

    additionalLabel = "FINAL"
    fileInfo = ""

    for i, (model, tag) in enumerate(modelClassParameterList):

        print(f"------------------------------------------------------------")
        print(f"Model {i}: {tag}")

        model.fit(X,y)

        if tag == "svm":
            model_filename = f"Trained_Models/{additionalLabel}_{tag}_model.pth"
            model.save(model_filename)
        else:
            model_filename = f"Trained_Models/{additionalLabel}_{tag}_model.joblib"
            joblib.dump(model, model_filename)
            print("Model saved!", model_filename)
        
        fileInfo += f'{model_filename},{tag}\n'

    print(f"------------------------------------------------------------")
    
    filename = "Trained_Models/Final_Base_Learners_Models.txt"
    with open(filename, "a") as f:
        f.write(fileInfo)
    
def trainBaseLearners(X, y):
    """
    This function trains multiple base learner models using cross-validation.
    It defines a list of models with their parameters and passes them to the
    MethodOfCrossValidation function for training and evaluation.

    This allows us to optimize and assess the performance of different models
    on the provided dataset (X, y).
    Args:
        X (pd.DataFrame): Feature dataset.
        y (pd.Series): Target labels.
    Returns:
        None
    """

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
        "n_estimators" : 80, 
        "criterion" : 'entropy', 
        "max_features" : 80,
        "random_state" : 42
    }

    svc_params = {
        "kernel" : "rbf",
        "C" : 20
    }

    modelClassParameterList = [
        # (RandomForestClassifier(**rf_params), "rf"), 
        # (KNeighborsClassifier(**knn_params), "knn"),
        # (XGBClassifier(**xgb_params), "xgb"),
        # (SVC(**svc_params), "svc"),
        # (ExtraTreesClassifier(**et_params), "et"),
        ]
    
    MethodOfCrossValidation(X, y, modelClassParameterList)
