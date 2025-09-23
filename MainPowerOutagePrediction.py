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
# from Models.LSTMModelFile import LSTMEstimator
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

    lstm_params = {"lstm_units": 64, 
                   "dropout": 0.5, 
                   "lr": 5e-4, 
                   "batch_size": 32}
    
    lstm_model = LSTMEstimator(**lstm_params)  # Placeholder for LSTM model instance

    baseLearners = [
            ('rf', RandomForestClassifier(n_estimators=400, random_state=42)),

            ('svc', SVC(kernel='rbf', C=20, probability=True)),

            ('knn', KNeighborsClassifier(n_neighbors=4, 
                                            weights='distance', 
                                            metric='manhattan')),
            # ('lstm', lstm_model)
        ]
    
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    parameters = {'estimators': baseLearners, 
                  'final_estimator': RandomForestClassifier(bootstrap=False, 
                                                            max_depth=30,
                                                            min_samples_leaf=1,
                                                            min_samples_split=2,
                                                            n_estimators=300), 
                    'passthrough': True, 
                    'cv': cv_inner}

    modelClassParameterList = [StackingClassifier(**parameters)]

    parameters_list = [parameters]

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

        adict = {}
        model = sklearnModel(model=model, parameters= adict)
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

    # newMethodUsingGridSearchCV(X, y)

    oldMethodOfCrossValidation(X, y)


    


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