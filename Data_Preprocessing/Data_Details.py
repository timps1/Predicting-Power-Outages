import sys
import pandas as pd

for i in range(1, len(sys.argv)):
    df = pd.read_csv(sys.argv[i])
    targetLabel= "outage_flag"
    print("-"*20)
    print(f"Filename: {sys.argv[i]}")
    print("Training set target distribution:\n", df[targetLabel].value_counts())
    print(f"Absolute difference: {abs(df[targetLabel].value_counts()[1] - df[targetLabel].value_counts()[0])}")
    print(f"Majority to minority ratio: {min(df[targetLabel].value_counts()[1], df[targetLabel].value_counts()[0]) / max(df[targetLabel].value_counts()[1], df[targetLabel].value_counts()[0])}")
    print("-"*20)