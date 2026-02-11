import pandas as pd
from pathlib import Path

# Directory containing the outage files
data_dir = Path("/Users/Zhuanz2/Desktop/Lecture 25/S2/COMPSCI 760/Team/COMPSCI 760/data/eaglei_outages")

# Match all outage files
files = sorted(data_dir.glob("eaglei_outages_*.csv"))

def clean_outage(path):
    """Clean a single outage CSV file"""
    df = pd.read_csv(path, dtype=str)

    # Convert time column
    df["run_start_time"] = pd.to_datetime(df["run_start_time"], errors="coerce")

    # Convert numeric columns
    df["fips_code"] = pd.to_numeric(df["fips_code"], errors="coerce").astype("Int64")
    df["sum"] = pd.to_numeric(df["sum"], errors="coerce").fillna(0).astype(int)

    # Strip whitespace from string columns
    df["county"] = df["county"].str.strip()
    df["state"] = df["state"].str.strip()

    # Add year column
    df["year"] = df["run_start_time"].dt.year

    return df

# Batch read and clean all outage files
all_outages = []
for f in files:
    print(f"Cleaning {f.name} ...")
    all_outages.append(clean_outage(f))

# Concatenate into one DataFrame
df_outage = pd.concat(all_outages, ignore_index=True)

# Save the cleaned dataset
df_outage.to_csv("eaglei_outages_2014_2022_clean.csv", index=False)

print("Cleaning and merging completed!")
print("Final dataset shape:", df_outage.shape)
print(df_outage.head())
