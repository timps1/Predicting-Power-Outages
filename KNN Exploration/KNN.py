import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Global random seed
SEED = 114514
np.random.seed(SEED)

prefix = "AKL"

# CSV file path
file_path = f"./{prefix}_Training_Data_normalised.csv"

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

def hyperparameter_search(X_train, y_train, param_grid):
    """Perform hyperparameter tuning with GridSearchCV"""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model, grid_search

def evaluate_model(model, X_test, y_test):
    """Train/test split evaluation of the best model"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

def save_grid_search_results(grid_search, X_train_scaled, X_test_scaled,
                             y_train, y_test, output_path=f"{prefix}_grid_search_results.csv"):
    results = pd.DataFrame(grid_search.cv_results_)
    
    results["rank_train_score"] = results["mean_train_score"].rank(ascending=False, method="min").astype(int)
    
    # Prepare arrays to hold confusion matrix components
    train_TP, train_TN, train_FP, train_FN = [], [], [], []
    test_TP, test_TN, test_FP, test_FN = [], [], [], []
    
    for i, row in results.iterrows():
        # Extract parameter combination
        params = row["params"]
        model = KNeighborsClassifier(**params)

        # Train on training set
        model.fit(X_train_scaled, y_train)

        # --- Train set predictions ---
        y_train_pred = model.predict(X_train_scaled)
        tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred, labels=[0, 1]).ravel()
        train_TP.append(tp)
        train_TN.append(tn)
        train_FP.append(fp)
        train_FN.append(fn)

        # --- Test set predictions ---
        y_test_pred = model.predict(X_test_scaled)
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()
        test_TP.append(tp)
        test_TN.append(tn)
        test_FP.append(fp)
        test_FN.append(fn)

    # Add confusion matrix components to DataFrame
    results["train_TP"] = train_TP
    results["train_TN"] = train_TN
    results["train_FP"] = train_FP
    results["train_FN"] = train_FN

    results["test_TP"] = test_TP
    results["test_TN"] = test_TN
    results["test_FP"] = test_FP
    results["test_FN"] = test_FN

    # Final columns to keep
    columns_to_keep = [
        "mean_fit_time", "std_fit_time", 
        "param_metric", "param_n_neighbors", "param_weights",
        "train_TP", "train_TN", "train_FP", "train_FN",
        "test_TP", "test_TN", "test_FP", "test_FN",
        "mean_test_score", "std_test_score", "rank_test_score",
        "mean_train_score", "std_train_score", "rank_train_score",
        "mean_test_accuracy", "mean_train_accuracy"
    ]

    # Add alias columns
    results["mean_test_accuracy"] = results["mean_test_score"]
    results["mean_train_accuracy"] = results["mean_train_score"]

    results = results[columns_to_keep]
    results.to_csv(output_path, index=False)
    print(f"Saved grid search results to {output_path}")

def run_knn_pipeline(file_path, param_grid, target="outage_flag", sample=True, sample_size=100000, test_size=0.2):
    """Complete pipeline with optional random sampling"""
    X, y = load_data(file_path, target)
    
    if sample and len(X) > sample_size:
        df_sample = pd.concat([X, y], axis=1).sample(n=sample_size, random_state=42)
        X = df_sample.drop(columns=[target])
        y = df_sample[target]
        print(f"Randomly sampled {sample_size} rows for training")
    else:
        print(f"Using full dataset ({len(X)} rows) for training")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Preprocess
    X_train_scaled = preprocess(X_train)
    X_test_scaled = preprocess(X_test)

    # Step 1: Hyperparameter tuning
    best_model, grid_search = hyperparameter_search(X_train_scaled, y_train, param_grid)

    # Step 2: Holdout evaluation
    evaluate_model(best_model, X_test_scaled, y_test)
    
    save_grid_search_results(grid_search, X_train_scaled, X_test_scaled, y_train, y_test)

param_grid = {
    "n_neighbors": list(range(1, 10)),    # test k from 1 to 10
    "weights": ["uniform", "distance"],   # voting weight
    "metric": ["euclidean", "manhattan", "minkowski"]  # distance metrics
}

# Default: random sample of 100000 rows
run_knn_pipeline(file_path, param_grid)