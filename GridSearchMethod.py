from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

def MethodUsingGridSearchCV(X, y):

    scoring = {
        "accuracy": metrics.make_scorer(metrics.accuracy_score),
        "f1": metrics.make_scorer(metrics.f1_score),
        'f1_weighted': metrics.make_scorer(metrics.f1_score),
        "precision" : metrics.make_scorer(metrics.precision_score),
        "recall" : metrics.make_scorer(metrics.recall_score),
        "roc_auc" : metrics.make_scorer(metrics.roc_auc_score),
        "r2" : metrics.make_scorer(metrics.r2_score)
    }
    
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    modelClassParameterList = [ExtraTreesClassifier(), KNeighborsClassifier(), RandomForestClassifier(), XGBClassifier()]
    
    parameter_grid_list = [
        {
            "n_estimators" : [80], 
            "criterion" : ['entropy'], 
            "max_features" : [50, 60, 70, 80]
        },
        {
            "n_neighbors": [3,4,5,6],
            "weights": ["uniform", "distance"],
            "metric": ["minkowski", "manhattan"]
        },
        {
            "n_estimators": [100, 150, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        },
        {
            "learning_rate": [0.1, 0.2, 0.3],
            "max_depth": [4, 6, 8],
            "n_estimators": [100, 150, 200],
            "subsample": [0.8, 1]
        }
    ]
    
    for i, model in enumerate(modelClassParameterList):
        print(f"Starting GridSearchCV for {model.__class__.__name__}...")

        grid = GridSearchCV(
            estimator=model,
            param_grid=parameter_grid_list[i],
            cv=cv_outer,
            n_jobs=2,
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
        for col in results_df.columns:
            if "mean" not in col:
                results_df = results_df.drop(columns=col)
        print(results_df)
        print(f"Grid search results saved to GridSearchCV_Results({model.__class__.__name__}).csv")