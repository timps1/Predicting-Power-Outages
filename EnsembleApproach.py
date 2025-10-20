import sys
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
import time
from CrossValidationMethod import MethodOfCrossValidation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import InputLineManagement as ilm
from Data_Preprocessing.SUS import applySUS
from Data_Preprocessing.smote_method import apply_smote 
from sklearn.ensemble import ExtraTreesClassifier


GLOBAL_RF_PRAMAS = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1
    }

GLOBAL_ET_PRAMAS = {
    "n_estimators": 80,
    "criterion" : 'entropy', 
    "max_features" : 120,
    "random_state": 42,
    "n_jobs": -1
}

GLOBAL_XGB_PRAMAS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1
}

GLOBAL_META_TAG = "et"

def trainingEnsemble(X, y):

    finalEstimatorTag = "et"
    passthrough = True

    global GLOBAL_ET_PRAMAS
    global GLOBAL_RF_PRAMAS
    global GLOBAL_XGB_PRAMAS

    inputModelsFile = open(ilm.getArg("MODELS-FILENAME-READ"), "r")
    modelsWithTags = inputModelsFile.read().strip().split("\n")
    inputModelsFile.close()

    modelFilenamesAndTags = [tuple(modelAndTag.strip().split(",")) for modelAndTag in modelsWithTags]

    X = loadBaseLearnersModel(X)

    meta_models = [(ExtraTreesClassifier(**GLOBAL_ET_PRAMAS), "et"), 
                   (RandomForestClassifier(**GLOBAL_RF_PRAMAS), "rf")
                   ]

    MethodOfCrossValidation(X, y, meta_models)

    

def fitAndSave(X, y):  
    global GLOBAL_META_TAG
    GLOBAL_META_TAG = "et"
    global GLOBAL_ET_PRAMAS
    global GLOBAL_RF_PRAMAS
    global GLOBAL_XGB_PRAMAS

    X = loadBaseLearnersModel(X)

    meta_model = RandomForestClassifier(**GLOBAL_ET_PRAMAS)

    X, y = applySUS(X, y)
    X, y = apply_smote(X, y)

    meta_model.fit(X, y)

    model_filename = f"Trained_Models/{GLOBAL_META_TAG}_ensemble_model.joblib"
    joblib.dump(meta_model, model_filename)
    print("Model saved!", model_filename)

def openAndPredict(X, y, filename = None):

    global GLOBAL_META_TAG

    passthrough = True

    inputModelsFile = open(ilm.getArg("MODELS-FILENAME-READ"), "r")
    modelsWithTags = inputModelsFile.read().strip().split("\n")
    inputModelsFile.close()

    modelFilenamesAndTags = [tuple(modelAndTag.strip().split(",")) for modelAndTag in modelsWithTags]

    X = loadBaseLearnersModel(X)
    
    if filename is None:
        filename = f"Trained_Models/{GLOBAL_META_TAG}_ensemble_model.joblib"

    model = joblib.load(filename)

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

def loadBaseLearnersModel(X):
    passthrough = True
    inputModelsFile = open(ilm.getArg("MODELS-FILENAME-READ"), "r")
    modelsWithTags = inputModelsFile.read().strip().split("\n")
    inputModelsFile.close()

    modelFilenamesAndTags = [tuple(modelAndTag.strip().split(",")) for modelAndTag in modelsWithTags]

    predictionsList = []
    start = time.time()

    for i, (modelFilename, tag) in enumerate(modelFilenamesAndTags):
        if tag == "svm":
            model = SVM()
            model.load(modelFilename)
        else: 
            model = joblib.load(modelFilename)
        
        if int(ilm.getArg("VERBOSE")) >= 1: print(f"\rProgress: {100 * i/len(modelFilenamesAndTags):.2f}% - {i} | ETA {(time.time() - start)/(i+1)*(len(modelFilenamesAndTags) - i):.2f}s | {tag}", end="", flush=True)
        
        predictionsList.append(model.predict_proba(X)[:, 1])
    
    print()
    prediction_cols = [
        pd.Series(preds, name=f"{modelFilenamesAndTags[i][1]} prediction")
        for i, preds in enumerate(predictionsList)
    ]

    if passthrough:
        X = pd.concat([X] + prediction_cols, axis=1)
    else:
        X = pd.concat(prediction_cols, axis=1)

    return X