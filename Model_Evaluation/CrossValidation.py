from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import pandas as pd
import os

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
            reports_list.append(report_df)

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
        print(avg_report)
        self.reports.append(avg_report)

        self.saveResults()
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        self.scores.append({
            "Model": self.model.getModelInfo(),
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