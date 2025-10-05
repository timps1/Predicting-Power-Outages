import sys
from Models.SklearnModels import sklearnModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from Model_Evaluation import CrossValidation
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
from Models.SVMtorch import SVM
import xgboost
from xgboost import XGBClassifier
import time

def MethodOfCrossValidation(X, y, passedModels = None):

    
    rf_params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1
    }

    svm_params = {
        "n_features" : 5000, 
        "gamma" : "scale", 
        "lr" : 0.01, 
        "weight_decay" : 0.0033333
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

    # baseLearnerParams = [rf_params, svm_params, knn_params, xgb_params]

    # # Base learners
    # baseLearners = [
    #     ('rf', RandomForestClassifier(**rf_params)),
    #     ('svc', SVC(**svm_params)),
    #     ('knn', KNeighborsClassifier(**knn_params)),
    #     ('xgb', XGBClassifier(**xgb_params)),
    # ]
    
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # parameters = {'estimators': baseLearners, 
    #               'final_estimator': RandomForestClassifier(bootstrap=False, 
    #                                                         max_depth=30,
    #                                                         min_samples_leaf=1,
    #                                                         min_samples_split=2,
    #                                                         random_state=42,
    #                                                         n_estimators=300), 
    #                 'passthrough': True, 
    #                 'cv': cv_inner,
    #                 'n_jobs' : -1,
    #                 'verbose' : 1
    #                 }

    if passedModels is None:
        modelClassParameterList = [
            # (RandomForestClassifier(**rf_params), "rf"), 
            # (KNeighborsClassifier(**knn_params), "knn"),
            # (XGBClassifier(**xgb_params), "xgb"),
            (SVM(**svm_params), "svm"),
            ]
    else:
        modelClassParameterList = passedModels
    # Cross-validation
    cross_validator = CrossValidation.CrossValidator(n_splits=5, storeResults=True)

    modelsList = []
    current_model_type = None

    for i, (model, tag) in enumerate(modelClassParameterList):

        if current_model_type is None:
            current_model_type = model.__class__.__name__

        elif current_model_type != model.__class__.__name__:
            current_model_type = model.__class__.__name__
            cross_validator.reset()  # Reset cross-validator for new model type

        model = sklearnModel(model=model)
        print(f"------------------------------------------------------------")
        print(f"Model {i}: {model.getModelInfo()}")

        cross_validator.storeResults = (i == len(modelsList) - 1)

        cross_validator.setModel(model)
        cross_validator.crossValidate(X.values, y.values)
        if tag == "xgb":
            
            print(f"Internal parameters: {model.model.get_xgb_params()}")
        
        print(f"parameters: {model.model.get_params()}")

        modelsList.append((model, tag))
        print(f"------------------------------------------------------------")
    
    cross_validator.printBestModel()
    return modelsList