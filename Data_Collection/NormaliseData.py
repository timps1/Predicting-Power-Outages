import numpy as np
import pandas as pd
import sys

def main():
    df = pd.read_csv(sys.argv[1])
    methodLabel='zero_to_one'
    normalizeDataByColumn(df,  methodLabel)
    dotIndex = sys.argv[1].rfind(".")
    nexFilename = sys.argv[1][:dotIndex] + "_normalised" + ".csv"
    df.to_csv(nexFilename, index=False)

def normalizeDataByColumn(data, methodLabel='zero_to_one'):
    """
    Normalize each column of data.
    """
    print("Processed columns:", end=" | ")
    for columnName in data.columns:
        print(columnName, end=" | ")
        if 'outage' in columnName:  # Skip target column
                continue
        elif "Humidity" in columnName:  # Skip Humidity columns
            continue
        elif "Time" in columnName:
            continue
        elif methodLabel == 'zero_to_one':
            normalizeDataZeroToOne(data, columnName)
        elif methodLabel == 'minus_one_to_one':
            normalizeDataMinusOneToOne(data, columnName)
        
        ######## Add more normalization methods with methodLabel as needed #########
        
        else:
            raise ValueError(f"Normalization method '{methodLabel}' is not supported." + 
                                "\nMake sure to add the method label normalizeData" +
                                " to the Data_Augmentation/NormaliseData.py file.")
        
    print()
def normalizeDataZeroToOne(data, columnName):
    min_val = np.min(data[columnName])
    max_val = np.max(data[columnName])
    if min_val != max_val:
        data[columnName] = (data[columnName] - min_val) / (max_val - min_val)
    

def normalizeDataMinusOneToOne(data, columnName):
    min_val = np.min(data[columnName], axis=0)
    max_val = np.max(data[columnName], axis=0)
    data[columnName] = 2 * (data[columnName] - min_val) / (max_val - min_val) - 1

###########################################################################
# Additional methods can be added here for other normalization techniques #
# Be sure to add method label normalizeData
###########################################################################


if __name__ == "__main__":
    main()