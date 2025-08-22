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

# Check if the script is being run as a standalone program
if __name__ == "__main__":
    print("Starting dataset split...")
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

# Split the dataset into training and testing sets
randomState = 42 # For reproducibility
testSize = 0.2  # 20% of the data will be used for testing
trainSize = 1 - testSize
targetLabel = 'target'  # Default target column name

if 'target' not in data.columns:
    print("The dataset must contain a 'target' column for splitting.")
    print("Available columns:", data.columns)
    targetLabel = input("Please enter the name of the target column: ")
    if targetLabel not in data.columns:
        print(f"Column '{targetLabel}' does not exist in the dataset.")
        sys.exit(1)

X = data.drop(targetLabel, axis=1)  # Assuming 'target' is the label column
y = data[targetLabel]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomState)

# Check distribution of target variable in training and testing sets
print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
print("Training set target distribution:\n", y_train.value_counts())
print("Testing set target distribution:\n", y_test.value_counts())

# Save the training and testing sets to new CSV files
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("Data split completed successfully.")
print("Training set saved to 'X_train.csv' and 'y_train.csv'.")
print("Testing set saved to 'X_test.csv' and 'y_test.csv'.")

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Dataset split completed successfully.")