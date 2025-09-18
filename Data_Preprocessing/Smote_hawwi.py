# ==========================
# 1. Import libraries
# ==========================
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ==========================
# 2. Load dataset
# ==========================
df = pd.read_csv("data/da/normalized_Hawaii_Training_Data_Cleaned.csv")   # adjust the path to your file

# Define features and target
X = df.drop(columns=["outage_flag"])
y = df["outage_flag"]

# ==========================
# 3. Train-test split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================
# 4. Apply SMOTE
# ==========================
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# ==========================
# 5. Show results
# ==========================
print("Original class distribution in training set:\n", y_train.value_counts())
print("\nClass distribution after SMOTE:\n", y_res.value_counts())
