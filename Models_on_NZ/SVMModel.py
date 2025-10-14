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

    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.model = svm.SVC(kernel=self.kernel, C=self.C)

    def getType(self):
        return "SVM"
