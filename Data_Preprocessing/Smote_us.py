# ==========================
# 1. Import libraries
# ==========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# ==========================
# 2. Load dataset
# ==========================
df = pd.read_csv("data/da/wwa_with_outages.csv", low_memory=False)

# Target variable
y = df["outage_flag"]

# Feature selection (remove time/ID-related columns)
X = df.drop(columns=[
    "outage_flag", "issued", "expired", "init_iss", "init_exp",
    "updated", "polybegin", "polyend", "product_id", "date"
], errors="ignore")

# ==========================
# 3. Encode categorical variables
# ==========================
cat_cols = X.select_dtypes(include=["object"]).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# ==========================
# 4. Handle missing values
# ==========================
# Numeric features: fill with mean
num_cols = X.select_dtypes(include=["float64", "int64"]).columns
if len(num_cols) > 0:
    num_imputer = SimpleImputer(strategy="mean")
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

# Categorical features: fill with mode
cat_cols = X.select_dtypes(include=["int32", "int", "object"]).columns
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# ==========================
# 5. Train-test split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================
# 6. Apply SMOTE oversampling
# ==========================
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# ==========================
# 7. Print results
# ==========================
print("Original class distribution in training set:\n", y_train.value_counts())
print("\nClass distribution after SMOTE:\n", y_res.value_counts())
