import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class sklearnModel:
    """
    Super class for other classes using sklearn to use shared functions

    Requires subclass to define model.
    """

    def train(self, X, y):
        """
        Train the SVM model.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        return accuracy
    
    def getModelInfo(self):
        """
        Get model information.
        """
        return str(self.model.__class__.__name__) + self.get_info()