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

if len(sys.argv) < 2:
    print("Usage: python TestTrainingSplit.py <data_file_path>")
    sys.exit(1)

fileName = sys.argv[1]
data = pd.read_csv(fileName)

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
positives = data[data[targetLabel] == 1]
negatives = data[data[targetLabel] == 0]

print(f"Total samples: {len(data)}")
print(f"Positive samples: {len(positives)}")
print(f"Negative samples: {len(negatives)}")

# --- Split positives: half to test, half to train ---
pos_train, pos_test = train_test_split(
    positives, 
    test_size=0.5, 
    random_state=randomState
)
if len(pos_train) < len(pos_test):
    pos_train, pos_test = pos_test, pos_train

# --- Split negatives normally to keep test proportion ~20% total ---
neg_train, neg_test = train_test_split(
    negatives, 
    test_size=0.5, 
    random_state=randomState
)

# --- Combine and shuffle ---
train_df = pd.concat([pos_train, neg_train], axis=0).sample(frac=1, random_state=randomState).reset_index(drop=True)
test_df = pd.concat([pos_test, neg_test], axis=0).sample(frac=1, random_state=randomState).reset_index(drop=True)

# --- Print distribution ---
print("Training set size:", len(train_df))
print("Testing set size:", len(test_df))
print("Training set target distribution:\n", train_df[targetLabel].value_counts())
print("Testing set target distribution:\n", test_df[targetLabel].value_counts())

# --- Save combined files ---
train_df.to_csv(f'{sys.argv[1]}_train_set.csv', index=False)
test_df.to_csv(f'{sys.argv[1]}_test_set.csv', index=False)

print("Data split completed successfully.")
print("Training set saved to 'train.csv'.")
print("Testing set saved to 'test.csv'.")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")