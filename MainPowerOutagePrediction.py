import sys
from Models.SklearnModels import sklearnModel
from IO_Data import IOData, IOModels
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from Model_Evaluation import CrossValidation
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
# from Models.LSTMModelFile import LSTMModel
import pandas as pd
from xgboost import XGBClassifier


def newMethodUsingGridSearchCV(X, y):

    scoring = {
        "accuracy": metrics.make_scorer(metrics.accuracy_score),
        "f1": metrics.make_scorer(metrics.f1_score)
    }

    baseLearners = [
            ('rf', RandomForestClassifier(n_estimators=400, random_state=42)),

            ('svc', SVC(kernel='rbf', C=20, probability=True)),

            ('knn', KNeighborsClassifier(n_neighbors=4, 
                                            weights='distance', 
                                            metric='manhattan')),

            ('xgb', XGBClassifier(learning_rate = 0.1,
                                  gamma=0, 
                                  max_depth=18,
                                  subsample = 1,
                                  random_state=42,
                                  use_label_encoder=False))
        ]
    
    cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    parameters = {'estimators': baseLearners, 'final_estimator': GradientBoostingClassifier(), 'passthrough': True, 'cv': cv_inner}

    modelClassParameterList = [SVC()]
    
    parameter_grid_list = [
        {
            'C': [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'kernel': ['poly'],
        }
    ]
    
    for i, model in enumerate(modelClassParameterList):
        print(f"Starting GridSearchCV for {model.__class__.__name__}...")

        grid = GridSearchCV(
            estimator=model,
            param_grid=parameter_grid_list[i],
            cv=cv_outer,
            n_jobs=-1,
            verbose=2, 
            scoring=scoring,
            refit='f1'  # refit using the f1 score
        )
        
        # ✅ Fit the GridSearchCV to your data
        grid.fit(X, y)
        
        # Access best parameters and CV results after fitting
        print(f"Best parameters found: {grid.best_params_}")
        print(f"Best cross-validation score: {grid.best_score_}")
        
        results_df = pd.DataFrame(grid.cv_results_)
        results_df.to_csv(f'GridSearchCV_Results({model.__class__.__name__}).csv', index=False)
        print(f"Grid search results saved to GridSearchCV_Results({model.__class__.__name__}).csv")


def oldMethodOfCrossValidation(X, y):
    baseLearners = [
            ('rf', RandomForestClassifier(n_estimators=400, random_state=42)),

            ('svc', SVC(kernel='rbf', C=60)),

            ('knn', KNeighborsClassifier(n_neighbors=4, 
                                            weights='distance', 
                                            metric='manhattan'))

        ]
    
    parameters = {'estimators': baseLearners, 'final_estimator': GradientBoostingClassifier(), 'passthrough': True}

    modelClassParameterList = [StackingClassifier(**parameters)]

    parameters_list = [
        {'n_neighbors' : 4, "weights" : 'distance', "metric" : 'manhattan'}, #KNN_parameters
        {'kernel' : 'rbf', 'C' : 60}, #SVM_parameters
        {'n_estimators' : 400, 'random_state' : 42}, #RF_parameters
        {'learning_rate' : 0.1, 'n_estimators' : 100, 'max_depth' : 20}, #XGB_parameters
        # {"lstm_units": 64, "dropout": 0.5, "lr": 5e-4, "batch_size": 32}  #LSTM_parameters
    ]
    modelClassParameterList = [
        
        KNeighborsClassifier(**parameters_list[0]),
        SVC(**parameters_list[1]),
        RandomForestClassifier(**parameters_list[2]),
        # LSTMModel(**parameters_list[4])
    ]

    # Cross-validation
    cross_validator = CrossValidation.CrossValidator(n_splits=10, storeResults=True)

    modelsList = []
    current_model_type = None

    for i, model in enumerate(modelClassParameterList):

        if current_model_type is None:
            current_model_type = model.__class__.__name__

        elif current_model_type != model.__class__.__name__:
            current_model_type = model.__class__.__name__
            cross_validator.reset()  # Reset cross-validator for new model type

        model = sklearnModel(model=model, parameters=parameters_list[i] if i < len(parameters_list) else {})
        print(f"------------------------------------------------------------")
        print(f"Model {i}: {model.getModelInfo()}")

        cross_validator.storeResults = (i == len(modelsList) - 1)

        cross_validator.setModel(model)
        print(f"Cross-validating model: {model.getModelInfo()}")
        cross_validator.crossValidate(X.values, y.values)

        modelsList.append(model)
        print(f"------------------------------------------------------------")
    
    cross_validator.printBestModel()

def main():
    """
    Main function to run the power outage prediction project.
    """

    # Get input data
    data = IOData.getInputData()
    
    if len(sys.argv) > 4:
        ############## Not sure if it works ##############
        # Load pre-trained model
        modelsRetrainList = IOModels.loadModel()
        print(f"Loaded model: {modelsRetrainList.__class__.__name__}")
        sys.exit(0)

    # Split data into features and target
    targetLabel = 'outage_flag'  # Assuming 'outage' is the target column
    if targetLabel not in data.columns:
        print(f"Column '{targetLabel}' does not exist in the dataset.")
        print("Available columns:", data.columns)
        sys.exit(1)
    
    X = data
    y = data[targetLabel]
    X = data.drop(columns=[targetLabel])  # features only

    if not check_for_non_numeric_values(X):
        print("Data contains non-numeric values. Please preprocess the data to convert all features to numeric types.")
        sys.exit(1)

    newMethodUsingGridSearchCV(X, y)

    # oldMethodOfCrossValidation(X, y)


    


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