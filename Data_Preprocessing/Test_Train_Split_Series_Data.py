"""
This script is used to split a dataset into training and testing sets.
Should really ideally be used only once. 
Otherwise, there is a risk of overfitting the model to the test set.

It will read a dataset from a CSV file, split it into training and testing sets,
and save the resulting datasets to new CSV files.
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Entry guard ---
if __name__ == "__main__":
    print("Starting dataset split (balanced positive case handling)...")
    print("--------------------------------------------------------------------")
else:
    print("This script is intended to be run as a standalone program.")
    sys.exit(1)

if len(sys.argv) < 3:
    print("Usage: python TestTrainingSplit.py <data_file_path> <already_split_data_file_path>")
    sys.exit(1)

fileName = sys.argv[1]
data = pd.read_csv(fileName)

oreginal_split = sys.argv[2]
splitData = pd.read_csv(oreginal_split)

if data.empty:
    print("The dataset is empty. Please provide a valid dataset.")
    sys.exit(1)

randomState = 42
targetLabel = 'outage_flag'

if targetLabel not in data.columns:
    print("The dataset must contain a 'target' column for splitting.")
    print("Available columns:", data.columns)
    targetLabel = input("Please enter the name of the target column: ")
    if targetLabel not in data.columns:
        print(f"Column '{targetLabel}' does not exist in the dataset.")
        sys.exit(1)

# Separate positive and negative samples
positives = splitData[splitData[targetLabel] == 1]
negatives = splitData[splitData[targetLabel] == 0]
splitPositives = splitData[splitData[targetLabel] == 1]
splitNegatives = splitData[splitData[targetLabel] == 0]

print(f"Total samples: {len(data)}")
print(f"Positive samples: {len(positives)}")
print(f"Negative samples: {len(negatives)}")
print(f"Positive split samples: {len(splitPositives)}")
print(f"Negative split samples: {len(splitNegatives)}")

# --- Convert to datetime ---
data["date"] = pd.to_datetime(data["date"], errors="coerce")
data["end_date"] = pd.to_datetime(data["end_date"], errors="coerce")
splitData["date"] = pd.to_datetime(splitData["date"], errors="coerce")

# --- Build training set (any overlap with splitData dates) ---
train_mask = pd.Series(False, index=data.index)
for date in splitData["date"].dropna().unique():
    train_mask |= (data["date"] <= date) & (data["end_date"] >= date)

train_df = data.loc[train_mask].copy()
test_df = data.loc[~train_mask].copy()


# --- Print distribution ---
print("Training set size:", len(train_df))
print("Testing set size:", len(test_df))
print("Training set target distribution:\n", train_df[targetLabel].value_counts())
print("Testing set target distribution:\n", test_df[targetLabel].value_counts())

# --- Save combined files ---
train_df.to_csv(f'{sys.argv[1][:sys.argv[1].rfind(".")]}_train_set.csv', index=False)
test_df.to_csv(f'{sys.argv[1][:sys.argv[1].rfind(".")]}_test_set.csv', index=False)

print("Data split completed successfully.")
print(f"Training set saved to '{sys.argv[1][:sys.argv[1].rfind(".")]}_train_set.csv'.")
print(f"Testing set saved to '{sys.argv[1][:sys.argv[1].rfind(".")]}_test_set.csv'.")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")