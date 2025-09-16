from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import pandas as pd

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

    def setModel(self, model):
        """
        Set the model to be used for cross-validation.
        """
        self.model = model

    def crossValidate(self, X, y):
        """
        Perform cross-validation on the model.
        """
        if self.model is None:
            raise ValueError("Model must be set before cross-validation.")
        
        foldNumber = 1
        folds_list = []
        scores_list = []
        reports_list = []

        for train_index, test_index in self.kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # train and predict
            self.model.train(X_train, y_train)
            y_pred = self.model.predict(X_test)

            # overall score
            score = self.model.evaluate(X_test, y_test)
            folds_list.append(foldNumber)
            scores_list.append(score)

            # classification report
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            report_df["Fold"] = foldNumber
            report_df["Model"] = self.model.getModelInfo()
            self.reports.append(report_df)
            reports_list.append(report_df)

            foldNumber += 1

        average_score = sum(scores_list) / len(scores_list)
        print(f">>>>>>>>>>>>>>>> Average Score across {self.n_splits} folds: {average_score}")

        average_report = pd.concat(reports_list).groupby('Model').mean().reset_index()
        print("Average Classification Report:")
        print(average_report)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        

        self.scores.append([folds_list, scores_list])
    
    def saveResults(self):
        """
        Save the cross-validation results to CSV files.
        """
        if not self.storeResults:
            print("Results storage is disabled.")
            return
        if self.model is None:
            print("No model has been set for cross-validation.")
            return
        
        try:
            
            # save detailed classification reports
            if self.reports:
                all_reports_df = pd.concat(self.reports, ignore_index=True)
                all_reports_df.to_csv(f'classification_reports({self.model.__class__.__name__}).csv', index=False)

            print(f"Results saved for model {self.model.__class__.__name__}.")
        
        except Exception as e:
            print(f"Error saving cross-validation results: {e}")

    def reset(self):
        """
        Reset the cross-validator state.
        """
        self.scores = []
        self.reports = []
        self.model = None