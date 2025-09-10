import sklearn
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Models.SklearnModels import sklearnModel

class XGBoost(sklearnModel):
    """
    A class to handle SVM model training and evaluation.
    """

    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = GradientBoostingClassifier(n_neighbors=self.n_neighbors, weights=self.weights)

    def getType(self):
        return "XGBoost"