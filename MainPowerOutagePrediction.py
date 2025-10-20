import sys
from IO_Data import IOData
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
from xgboost import XGBClassifier
from CrossValidationMethod import MethodOfCrossValidation
from GridSearchMethod import MethodUsingGridSearchCV
from EnsembleApproach import trainingEnsemble, openAndPredict, fitAndSave
from BaseLearnersTraining import trainBaseLearners, buildAndSaveBaseLearners
from Data_Preprocessing import SUS
import InputLineManagement as ilm
import joblib

def main():
    """
    Main function to run the power outage prediction project.
    """

    # Reading inputline
    ilm.intialiseGLOBAL_DICTIONARY()

    # Get input data
    data = pd.read_csv(ilm.getArg("INPUT"))

    if ilm.getArg("SECONDARY-INPUT") != 0:
        data = pd.concat([data,pd.read_csv(ilm.getArg("SECONDARY-INPUT"))], axis=0)
    
    # Split data into features and target
    targetLabel = 'outage_flag'  # Assuming 'outage' is the target column
    
    for columnName in data.columns:
        if "date" in columnName or "Time_" in columnName or "Unnamed" in columnName:
            data = data.drop(columns=columnName)

    X = data
    y = data[targetLabel]
    # for col in data.columns:
    #     if "hail" in col.lower(): #or "thunder" in col.lower():
    #         data = data.drop(columns=[col])
    X = data.drop(columns=[targetLabel])  # features only

    # X = pd.concat(X, y, axis=1)
    # X.to_csv(f"{sys.argv[1][:sys.argv[1].rfind(".")]}_near_miss.csv")

    # print("Data columns:", data.columns)
    if not check_for_non_numeric_values(X):
        print("Data contains non-numeric values. Please preprocess the data to convert all features to numeric types.")
        sys.exit(1)


    # buildAndSaveBaseLearners(X, y)

    if ilm.getArg("ENSEMBLE-INPUT") is not None:
        data = pd.read_csv(ilm.getArg("ENSEMBLE-INPUT"))
        X = data
        for columnName in data.columns:
            if "date" in columnName or "Time_" in columnName or "Unnamed" in columnName:
                data = data.drop(columns=columnName)
        y = data[targetLabel]
        X = data.drop(columns=[targetLabel]) 

        fitAndSave(X, y)

    if ilm.getArg("TEST-SET") is not None:
        data = pd.read_csv(ilm.getArg("TEST-SET"))
        X = data
        for columnName in data.columns:
            if "date" in columnName or "Time_" in columnName or "Unnamed" in columnName:
                data = data.drop(columns=columnName)
        y = data[targetLabel]
        X = data.drop(columns=[targetLabel]) 
        openAndPredict(X, y)
    



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
    main()
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Power outage prediction project completed successfully.")