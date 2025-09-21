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

            ('xgb', XGBClassifier(learning_rate = 0.5,
                                  gamma=0, 
                                  max_depth=18,
                                  subsample = 1,
                                  random_state=42)),
        ]
    
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    parameters = {'estimators': baseLearners, 'final_estimator': GradientBoostingClassifier(), 'passthrough': True, 'cv': cv_inner}

    modelClassParameterList = [StackingClassifier(**parameters), StackingClassifier(**parameters)]
    
    parameter_grid_list = [

        # {
        #     'final_estimator': [XGBClassifier()],
        #     'final_estimator__n_estimators': [100, 300],
        #     'final_estimator__learning_rate': [0.01, 0.05, 0.1],
        #     'final_estimator__max_depth': [3, 5, 7],
        #     'final_estimator__gamma': [0, 5],
        #     'passthrough': [True, False]
        # },

        {
            'final_estimator': [KNeighborsClassifier()],
            'final_estimator__n_neighbors': [3, 5, 7, 9],
            'final_estimator__weights': ['uniform', 'distance'],
            'final_estimator__metric': ['euclidean', 'manhattan'],
            'passthrough': [True, False]
        },

        {
            'final_estimator': [RandomForestClassifier()],
            'final_estimator__n_estimators': [100, 300],
            'final_estimator__max_depth': [None, 10, 30],
            'final_estimator__min_samples_split': [2, 10],
            'final_estimator__min_samples_leaf': [1, 4],
            'final_estimator__bootstrap': [True, False],
            'passthrough': [True, False]
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
        try:
            results_df.to_csv(f'GridSearchCV_Results({model.__class__.__name__})({i}).csv', index=False)
            print(f"Grid search results saved to GridSearchCV_Results({model.__class__.__name__})({i}).csv")
        except Exception as e:
            print("Failed", e)
            results_df.to_csv(f'GridSearchCV_Results(Unknown).csv', index=False)
            print(f"Grid search results saved to GridSearchCV_Results(Unknown).csv")



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