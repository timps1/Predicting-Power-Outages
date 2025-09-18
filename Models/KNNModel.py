import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Models.SklearnModels import sklearnModel


class KNN(sklearnModel):
    """
    A class to handle SVM model training and evaluation.
    """

    def __init__(self, parameters=(5, 'uniform')):
        self.n_neighbors = parameters[0]
        self.weights = parameters[1]
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)

    def getType(self):
        return "KNN"
    
    def get_info(self):
        return f"(n_neighbors_{self.n_neighbors})(weights_{self.weights})"
    
