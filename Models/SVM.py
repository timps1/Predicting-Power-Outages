import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SVMModel:
    """
    A class to handle SVM model training and evaluation.
    """

    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.model = svm.SVC(kernel=self.kernel, C=self.C)

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
    
