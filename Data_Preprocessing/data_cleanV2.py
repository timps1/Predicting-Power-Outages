import pandas as pd
import numpy as np

# Load your CSV
df = pd.read_csv("Hawaii_Training_Data.csv")

# 1. Replace '%' and 'mbar' and other non-numeric chars, then convert to float
df = df.replace(r'[^0-9\.-]', '', regex=True)

# 2. Convert everything to numeric, forcing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# 3. Fill missing (NaN) with 0
df = df.fillna(0.0)

cols_to_remove = ["Time_0", "Time_1", "Time_2", "Time_3"]  # your list here
for col in df.columns:
    if col.startswith("Time_") and col not in cols_to_remove:
        cols_to_remove.append(col)
df = df.drop(columns=cols_to_remove, errors="ignore")  # ignore avoids errors if not found

print(df.head())

df.to_csv("Hawaii_Training_Data_Cleaned.csv", index=False)