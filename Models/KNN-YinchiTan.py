import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Global random seed
SEED = 114514
np.random.seed(SEED)

# CSV file path
file_path = "..\wwa_with_outages_detailed.csv"

# Read first 5 rows
df_head = pd.read_csv(file_path, nrows=5)
print("First 5 rows:")
print(df_head)

# Get total number of rows (excluding header)
total_rows = sum(1 for _ in open(file_path, encoding="utf-8")) - 1
print(f"Total rows: {total_rows}")

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

def train_and_evaluate_knn(X, y, k_values=range(1, 11)):
    """Train KNN with train/test split and 10-fold cross-validation"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_acc = cross_val_score(knn, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        cv_f1 = cross_val_score(knn, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1).mean()

        print(
            f"k={k}, "
            f"Holdout: ACC={acc:.4f}, F1={f1:.4f} | "
            f"10-CV: ACC={cv_acc:.4f}, F1={cv_f1:.4f}"
        )

def run_knn_pipeline(file_path, target="outage_flag", sample=True, sample_size=100000):
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
    train_and_evaluate_knn(X_scaled, y)

# Default: random sample of 100000 rows
run_knn_pipeline(file_path)