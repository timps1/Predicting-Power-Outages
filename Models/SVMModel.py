import sklearn
from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Models.SklearnModels import sklearnModel


class SVM(sklearnModel):
    """
    A class to handle SVM model training and evaluation.
    """

    def __init__(self, parameters=('linear', 1.0)):
        self.kernel = parameters[0]
        self.C = parameters[1]
        self.model = svm.SVC(kernel=self.kernel, C=self.C)

    def getType(self):
        return "SVM"
    
    def get_info(self):
        return f"(kernel_{self.kernel})(C_{self.C})"
