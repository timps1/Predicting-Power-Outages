from .. import Models
from sklearn.model_selection import KFold
import pandas as pd

class CrossValidator:
    """
    A class to handle cross-validation for model evaluation.
    """

    def __init__(self, n_splits=10, storeResults=False):
        self.model = None
        self.n_splits = n_splits
        self.kf = KFold(n_splits=self.n_splits, shuffle=True)
        self.scores = []
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
        for train_index, test_index in self.kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            self.model.train(X_train, y_train)
            score = self.model.evaluate(X_test, y_test)
            self.scores.append([foldNumber, score])
            foldNumber += 1

        if self.storeResults:
            self.save_results()

    
    def saveResults(self):
        """
        Save the cross-validation results to a file.
        """
        if not self.storeResults:
            print("Results storage is disabled.")
            return
        if self.model is None:
            print("No model has been set for cross-validation.")
            return
        
        try:
            results_df = pd.DataFrame(self.scores, columns=['Fold', 'Accuracy'])
            results_df.to_csv(f'cross_validation_results({self.model.__class__.__name__}).csv', index=False)
            print(f"Cross-validation results saved to 'cross_validation_results({self.model.__class__.__name__}).csv'.")
        except Exception as e:
            print(f"Error saving cross-validation results: {e}")
            try:
                print("Attempting to save results in a text format...")
                resultsString = "\n".join([f"Fold {fold}: Accuracy {score}" for fold, score in self.scores])
                with open(f'cross_validation_results({self.model.__class__.__name__}).txt', 'w') as f:
                    f.write(resultsString)
                print(f"Cross-validation results saved to 'cross_validation_results({self.model.__class__.__name__}).txt'.")
            except Exception as e:
                print(f"Failed to save results in text format: {e}")
                return