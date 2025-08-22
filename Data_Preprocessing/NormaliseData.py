import numpy as np
import pandas as pd

class DataNormalizer:
    """
    A class to normalize data using different methods.
    """

    def __init__(self):
        pass

    def normalizeDataByColumn(self, data, methodLabel='zero_to_one'):
        """
        Normalize each column of data.
        """

        for columnName in data.columns:
            if methodLabel == 'zero_to_one':
                self.normalizeDataZeroToOne(data, columnName)
            elif methodLabel == 'minus_one_to_one':
                self.normalizeDataMinusOneToOne(data, columnName)
            
            ######## Add more normalization methods with methodLabel as needed #########
            
            else:
                raise ValueError(f"Normalization method '{methodLabel}' is not supported." + 
                                 "\nMake sure to add the method label normalizeData" +
                                 " to the Data_Augmentation/NormaliseData.py file.")
            
    @staticmethod
    def normalizeDataZeroToOne(data, columnName):
        min_val = np.min(data[columnName])
        max_val = np.max(data[columnName])
        data[columnName] = (data[columnName] - min_val) / (max_val - min_val)
        

    @staticmethod
    def normalizeDataMinusOneToOne(data, columnName):
        min_val = np.min(data[columnName], axis=0)
        max_val = np.max(data[columnName], axis=0)
        data[columnName] = 2 * (data[columnName] - min_val) / (max_val - min_val) - 1

    ###########################################################################
    # Additional methods can be added here for other normalization techniques #
    # Be sure to add method label normalizeData
    ###########################################################################