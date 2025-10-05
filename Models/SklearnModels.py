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
    def __init__(self, model=None, parameters=None):
        self.model = model  # Placeholder for the sklearn model instance
        self.parameters = model.get_params()  # Placeholder for model parameters

    def fit(self, X, y):
        """
        Fit the model to the training data.
        """
        self.model.fit(X, y)

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
        aString = str(self.model.__class__.__name__)
        for key, value in self.parameters.items():
            value = self.processParameters(value)
            aString += f"({key}_{value})"

        return aString
    
    def processParameters(self, value):
        """
        Process and set model parameters.
        """
        if isinstance(value, float):
            value = round(value, 4)
        elif isinstance(value, str):
            value = value.replace(" ", "_")
        elif isinstance(value, dict):
            value = len(value)  # just show number of items in dict
        elif isinstance(value, tuple):
            aString = value[0]
        elif isinstance(value, list):
            aString = ""
            for i, v in enumerate(value):
                aString += self.processParameters(v) + ("_" if i < len(value) - 1 else "")
            value = aString
        return value