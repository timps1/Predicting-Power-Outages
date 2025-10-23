import sys
import pandas as pd
from xgboost import XGBClassifier
from CrossValidationMethod import MethodOfCrossValidation
from GridSearchMethod import MethodUsingGridSearchCV
from EnsembleApproach import trainingEnsemble, openAndPredict, fitAndSave
from BaseLearnersTraining import trainBaseLearners, buildAndSaveBaseLearners
import time 

def main():
    """
    Main function to run the power outage prediction project.
    """

    if len(sys.argv) < 4:
        print("Usage: python MainPowerOutagePrediction.py <base_learner_data.csv> <ensemble_training_data.csv> <test_data.csv>")
        sys.exit(1)
    
    targetLabel = 'outage_flag'  # Assuming 'outage' is the target column
    
    X, y = loadAndSplitData(sys.argv[1], targetLabel)
    buildAndSaveBaseLearners(X, y) 

    X, y = loadAndSplitData(sys.argv[2], targetLabel)
    fitAndSave(X, y)

    X, y = loadAndSplitData(sys.argv[3], targetLabel)
    openAndPredict(X, y)

def loadAndSplitData(filename, targetLabel):
    data = pd.read_csv(filename)
    
    for columnName in data.columns:
        if "date" in columnName or "Time_" in columnName or "Unnamed" in columnName or "hail" in columnName:
            data = data.drop(columns=columnName)

    X = data
    y = data[targetLabel]
    X = data.drop(columns=[targetLabel])  # features only

    if not check_for_non_numeric_values(X):
        print("Data contains non-numeric values. Please preprocess the data to convert all features to numeric types.")
        sys.exit(1)
    return X, y

def check_for_non_numeric_values(df):
    """
    Checks for any values in the DataFrame that are not real numbers (int or float).
    """
    non_numeric_rows = {}
    
    for col in df.columns:
        mask = ~df[col].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))
        if mask.any():
            non_numeric_rows[col] = df[col][mask].unique()
    
    if non_numeric_rows:
        print("Found non-numeric values in columns:")
        for col, vals in non_numeric_rows.items():
            print(f"{col}: {vals}")
        return False
    else:
        print("All values are numeric.")
        return True

if __name__ == "__main__":
    print("Starting power outage prediction project...")
    print("--------------------------------------------------------------------")
    strartTime = time.time()
    main()
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Power outage prediction project completed successfully. Runtime: {round(time.time() - strartTime, 2)} seconds.")