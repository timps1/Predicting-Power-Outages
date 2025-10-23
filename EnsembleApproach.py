import pandas as pd
from xgboost import XGBClassifier
import joblib
import time
from CrossValidationMethod import MethodOfCrossValidation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Data_Preprocessing.SUS import applySUS
from Data_Preprocessing.smote_method import apply_smote 
from sklearn.ensemble import ExtraTreesClassifier

GLOBAL_ET_PRAMAS = {
    "n_estimators": 80,
    "criterion" : 'entropy', 
    "max_features" : 120,
    "random_state": 42,
    "n_jobs": -1
}

GLOBAL_META_TAG = "et"

def trainingEnsemble(X, y):

    global GLOBAL_ET_PRAMAS

    X = loadBaseLearnersModel(X)

    meta_models = [(ExtraTreesClassifier(**GLOBAL_ET_PRAMAS), "et")]

    MethodOfCrossValidation(X, y, meta_models)

    

def fitAndSave(X, y):  
    global GLOBAL_META_TAG
    global GLOBAL_ET_PRAMAS

    X = loadBaseLearnersModel(X)

    meta_model = ExtraTreesClassifier(**GLOBAL_ET_PRAMAS)

    X, y = applySUS(X, y, 0.2)
    X, y = apply_smote(X, y, 1)

    meta_model.fit(X, y)

    model_filename = f"Trained_Models/{GLOBAL_META_TAG}_Final_Ensemble_Model.joblib"
    joblib.dump(meta_model, model_filename)
    print("Model saved!", model_filename)

def openAndPredict(X, y, filename = None):

    global GLOBAL_META_TAG

    X = loadBaseLearnersModel(X)
    
    filename = f"Trained_Models/{GLOBAL_META_TAG}_Final_Ensemble_Model.joblib"

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
    inputModelsFile = open("Trained_Models/Final_Base_Learners_Models.txt", "r")
    modelsWithTags = inputModelsFile.read().strip().split("\n")
    inputModelsFile.close()

    modelFilenamesAndTags = [tuple(modelAndTag.strip().split(",")) for modelAndTag in modelsWithTags]

    predictionsList = []
    start = time.time()

    for i, (modelFilename, tag) in enumerate(modelFilenamesAndTags):
        model = joblib.load(modelFilename)
        
        print(f"\rProgress: {100 * i/len(modelFilenamesAndTags):.2f}% - {i} | {tag}", end="", flush=True)
        
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