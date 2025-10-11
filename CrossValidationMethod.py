import sys
from Models.SklearnModels import sklearnModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
from Models.SVMtorch import SVM
import xgboost
from xgboost import XGBClassifier
import time
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import os
from Data_Preprocessing.SUS import applySUS
from Data_Preprocessing.smote_method import apply_smote
import InputLineManagement as ilm



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
    cross_validator = CrossValidator(n_splits=5, storeResults=True)

    modelsList = []
    current_model_type = None

    for i, (model, tag) in enumerate(modelClassParameterList):

        if current_model_type is None:
            current_model_type = model.__class__.__name__

        elif current_model_type != model.__class__.__name__:
            current_model_type = model.__class__.__name__
            cross_validator.reset()  # Reset cross-validator for new model type

        if i > 0:
            cross_validator.remakeFolds = False

        model = sklearnModel(model=model)
        print(f"------------------------------------------------------------")
        print(f"Model {i}: {model.getModelInfo()}")

        cross_validator.storeResults = (i == len(modelsList) - 1)

        cross_validator.setModel(model)
        cross_validator.crossValidate(X.values, y.values, X.columns)
        if tag == "xgb":
            
            print(f"Internal parameters: {model.model.get_xgb_params()}")
        
        print(f"parameters: {model.model.get_params()}")

        modelsList.append((model, tag))
        print(f"------------------------------------------------------------")
    
    cross_validator.printBestModel()
    return modelsList


class CrossValidator:
    """
    A class to handle cross-validation for model evaluation.
    """

    def __init__(self, n_splits=10, storeResults=False):
        self.model = None
        self.n_splits = n_splits
        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.scores = []   # overall scores
        self.reports = []  # detailed classification reports
        self.storeResults = storeResults
        self.remakeFolds = int(ilm.getArg("RM-FOLDS"))==1 #Saved folds can be rerun

    def setModel(self, model):
        """
        Set the model to be used for cross-validation.
        """
        self.model = model


    def crossValidate(self, X, y, columnNames):
        """
        Perform cross-validation on the model.
        """
        if self.model is None:
            raise ValueError("Model must be set before cross-validation.")
        
        
        foldNumber = 1
        folds_list = []
        scores_list = []
        reports_list = []
        foldFileDir = "../"

        for train_index, test_index in self.kf.split(X):
            startTime = time.time()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if self.remakeFolds:
                X_train, y_train = applySUS(X_train, y_train)
                X_train, y_train = apply_smote(X_train, y_train)
                saveFoldDF = pd.concat([pd.DataFrame(X_train, columns=columnNames),
                                        pd.Series(y_train, name="outage_flag")], 
                                        axis=1)
                if ilm.getArg("SAVE-RM-FOLDS") == 1:
                    saveFoldDF.to_csv(foldFileDir+f"fold{foldNumber}.csv", index=False)
                    print(f"Saved fold csv: {foldFileDir}fold{foldNumber}.csv")
            
            else:
                saveFoldDF = pd.read_csv(foldFileDir+f"fold{foldNumber}.csv")
                X_train =  saveFoldDF.drop(columns="outage_flag")
                y_train = saveFoldDF["outage_flag"]
                X_train = X_train.values
                y_train = y_train.values

            # train and predict
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            # overall score
            score = accuracy_score(y_pred, y_test)
            folds_list.append(foldNumber)
            scores_list.append(score)

            # classification report
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            report_df["Fold"] = foldNumber
            report_df["Model"] = self.model.getModelInfo()
            reports_list.append(report_df)
            
            print(f"Finished fold {foldNumber} in {round(time.time() - startTime, 1)}s\n")
            foldNumber += 1

        # ---- Average accuracy ----
        average_score = sum(scores_list) / len(scores_list)
        print(f">>>>>>>>>>>>>>>> Average Score across {self.n_splits} folds: {average_score}")

        # ---- Average classification report ----
        all_reports = pd.concat(reports_list)

        # Drop non-numeric before averaging
        avg_report = (
            all_reports
            .drop(columns=["Fold", "Model", "support"], errors="ignore")
            .groupby(all_reports.index)
            .mean(numeric_only=True)
        )

        # Keep model name as a column
        model_name = reports_list[0]["Model"].iloc[0]
        avg_report.insert(0, "Model", model_name)

        print("Average Classification Report:")
        print(f"Model {avg_report["Model"][0]}")
        print(avg_report.drop(columns="Model").to_string())
        self.reports.append(avg_report)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        self.scores.append({
            "Model": self.model,
            "Average_Accuracy": average_score
        })

    def saveResults(self): 
        """ Save the cross-validation results to CSV files. """
        
        if self.model is None: 
            print("No model has been set for cross-validation.") 
            return 
        try:
            # --- Save detailed classification reports ---
            if self.reports:
                last_report_df = self.reports[-1]
                try:
                    reports_file = f'classification_reports({self.model.model.__class__.__name__}).csv'
                except:
                    reports_file = f'classification_reports({self.model.__class__.__name__}).csv'
                if os.path.exists(reports_file):
                    existing_df = pd.read_csv(reports_file)
                    all_reports_df = pd.concat([existing_df, last_report_df], ignore_index=True)
                else:
                    all_reports_df = last_report_df

                all_reports_df.to_csv(reports_file, index=False)

            print(f"✅ Results saved/appended for model {self.model.__class__.__name__}.")

        except Exception as e:
            print(f"Error saving cross-validation results: {e}")

    def reset(self):
        """
        Reset the cross-validator state.
        """
        self.scores = []
        self.reports = []
        self.model = None

    def printBestModel(self):
        """
        Print the model with the best average accuracy across cross-validation.
        """
        if not self.scores:
            print("No scores available. Run crossValidate first.")
            return
        
        scores_df = pd.DataFrame(self.scores)
        best_row = scores_df.loc[scores_df["Average_Accuracy"].idxmax()]

        print("\n🏆 Best Model:")
        print(best_row.to_frame().T)   # prints in table style