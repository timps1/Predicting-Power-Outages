import sys
from Models.SklearnModels import sklearnModel
from IO_Data import IOData
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from xgboost import XGBClassifier

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

    baseLearners = [
            ('rf', RandomForestClassifier(n_estimators=400, random_state=42)),

            ('svc', SVC(kernel='rbf', C=20, probability=True)),

            ('knn', KNeighborsClassifier(n_neighbors=2, 
                                            weights='distance', 
                                            metric='manhattan'
                                            )),

            ('xgb', XGBClassifier(learning_rate = 0.1,
                                  gamma=0, 
                                  max_depth=18,
                                  subsample = 1,
                                  random_state=42,
                                  use_label_encoder=False
                                  ))         
        ]
    
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    parameters = {'estimators': baseLearners, 'final_estimator': GradientBoostingClassifier(), 'passthrough': True, 'cv': cv_inner}

    modelClassParameterList = [KNeighborsClassifier()]
    
    parameter_grid_list = [
        # {
        #     'C': [10, 20, 30, 40, 50, 60],
        #     'kernel': ['rbf'],
        # },
        {
            "n_neighbors":[1,2,3,4], 
            "weights":['distance', "uniform"], 
            "metric":['manhattan', "euclidean"]
        },
        # {
        #     "n_estimators": [300, 400],
        #     "random_state": [42],
        #     "n_jobs": [-1]
        # }
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
        for col in results_df.columns:
            if "mean" not in col:
                results_df = results_df.drop(columns=col)
        print(results_df)
        print(f"Grid search results saved to GridSearchCV_Results({model.__class__.__name__}).csv")