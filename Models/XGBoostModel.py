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

    def __init__(self, parameters=(0.1, 100)):
        self.learning_rate= parameters[0] 
        self.n_estimators= parameters[1]
        self.model = GradientBoostingClassifier(learning_rate=self.learning_rate, 
                                                n_estimators = self.n_estimators)

    def getType(self):
        return "XGBoost"
    
    def get_info(self):
        return f"(n_neighbors_{self.n_neighbors})(weights_{self.weights})"