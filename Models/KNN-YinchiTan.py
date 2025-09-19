import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Global random seed
SEED = 114514
np.random.seed(SEED)

# CSV file path
file_path = "./normalized_Hawaii_Training_Data_Cleaned.csv"

'''
# Read first 5 rows
df_head = pd.read_csv(file_path, nrows=5)
print("First 5 rows:")
print(df_head)

# Get total number of rows (excluding header)
total_rows = sum(1 for _ in open(file_path, encoding="utf-8")) - 1
print(f"Total rows: {total_rows}")
'''

def load_data(file_path, target="outage_flag"):
    """Load CSV and separate features and target"""
    df = pd.read_csv(file_path)
    X = df[[col for col in df.columns if not col.startswith("outage_") and col != target]]
    y = df[target]
    return X, y

def handle_missing_values(X):
    """Fill missing values: numeric → median, categorical → mode"""
    X_filled = X.copy()
    for col in X_filled.columns:
        if X_filled[col].dtype in ["float64", "int64"]:
            X_filled.fillna({col: X_filled[col].median()}, inplace=True)
        else:
            X_filled.fillna({col: X_filled[col].mode()[0]}, inplace=True)
    return X_filled

def preprocess(X):
    """Missing value handling + categorical encoding + standardization"""
    X_clean = handle_missing_values(X)
    
    # Encode categorical variables
    for col in X_clean.select_dtypes(include=["object"]).columns:
        X_clean[col] = LabelEncoder().fit_transform(X_clean[col].astype(str))

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    return X_scaled

def hyperparameter_search(X, y, param_grid):
    """Perform hyperparameter tuning with GridSearchCV"""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        KNeighborsClassifier(n_jobs=-1),
        param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X, y)

    print("\nBest parameters found by GridSearchCV:")
    print(grid.best_params_)
    print(f"Best CV F1 score: {grid.best_score_:.4f}")
    
    # Sort all results by mean test score (descending)
    results = pd.DataFrame(grid.cv_results_)
    results = results.sort_values(by="mean_test_score", ascending=False)

    print("\nTop 10 parameter settings (by CV F1 score):")
    top10 = results.head(10)
    for i, row in top10.iterrows():
        print(
            f"Rank {row['rank_test_score']}: "
            f"params={row['params']}, "
            f"mean F1={row['mean_test_score']:.4f}, "
            f"std F1={row['std_test_score']:.4f}"
        )
        
    # Save Top 10 results to CSV
    top10.to_csv("knn_gridsearch_top10.csv", index=False)
    print("\nTop 10 results saved to knn_gridsearch_top10.csv")

    return grid.best_estimator_

def evaluate_model(model, X, y):
    """Train/test split evaluation of the best model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print("\nHoldout evaluation:")
    print(f"Accuracy: {acc:.4f}, F1 score: {f1:.4f}")

def run_knn_pipeline(file_path, param_grid, target="outage_flag", sample=True, sample_size=100000):
    """Complete pipeline with optional random sampling"""
    X, y = load_data(file_path, target)
    
    if sample and len(X) > sample_size:
        df_sample = pd.concat([X, y], axis=1).sample(n=sample_size, random_state=42)
        X = df_sample.drop(columns=[target])
        y = df_sample[target]
        print(f"Randomly sampled {sample_size} rows for training")
    else:
        print(f"Using full dataset ({len(X)} rows) for training")
    
    X_scaled = preprocess(X)

    # Step 1: Hyperparameter tuning
    best_model = hyperparameter_search(X_scaled, y, param_grid)

    # Step 2: Holdout evaluation
    evaluate_model(best_model, X_scaled, y)

param_grid = {
    "n_neighbors": list(range(1, 10)),    # test k from 1 to 10
    "weights": ["uniform", "distance"],   # voting weight
    "metric": ["euclidean", "manhattan", "minkowski"]  # distance metrics
}

# Default: random sample of 100000 rows
run_knn_pipeline(file_path, param_grid)