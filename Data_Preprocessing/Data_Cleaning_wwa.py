import pandas as pd
from pathlib import Path

# ==== 1. Directory containing the files ====
data_dir = Path("/Users/Zhuanz2/Desktop/Lecture 25/S2/COMPSCI 760/Team/COMPSCI 760/data/VTEC 2014-2022")

# Match all yearly files
files = sorted(data_dir.glob("wwa_*.csv"))

print("Number of files found:", len(files))
for f in files:
    print("✔", f.name)

# ==== 2. Define cleaning function ====
def clean_wwa(path):
    df = pd.read_csv(path, dtype=str)

    # Convert datetime columns
    time_cols = ["ISSUED", "EXPIRED", "INIT_ISS", "INIT_EXP", "UPDATED", "POLYBEGIN", "POLYEND"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert numeric columns
    num_cols = ["ETN", "AREA_KM2", "WINDTAG", "HAILTAG"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert boolean column
    if "IS_EMERGENCY" in df.columns:
        df["IS_EMERGENCY"] = df["IS_EMERGENCY"].astype(bool)

    # Remove duplicates based on PRODUCT_ID
    if "PRODUCT_ID" in df.columns:
        df = df.drop_duplicates(subset=["PRODUCT_ID"])

    # Clean string columns (strip whitespace, convert to uppercase)
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip().str.upper()

    return df

# ==== 3. Clean and merge all files ====
all_dfs = []
for f in files:
    print(f"Cleaning {f.name} ...")
    all_dfs.append(clean_wwa(f))

df_all = pd.concat(all_dfs, ignore_index=True)

# ==== 4. Add WFO -> STATE mapping ====
lookup = pd.read_csv(
    "/Users/Zhuanz2/Desktop/Lecture 25/S2/COMPSCI 760/Team/COMPSCI 760/data/wfo_lookup_table.csv",
    encoding="latin-1"   # or "cp1252"
)

# Normalize column names (remove extra whitespace)
lookup.columns = lookup.columns.str.strip()

# Merge: WWA's WFO corresponds to lookup's Office call sign
df_all = df_all.merge(
    lookup[["Office call sign", "State"]],
    left_on="WFO",
    right_on="Office call sign",
    how="left"
)

# Drop redundant column
df_all = df_all.drop(columns=["Office call sign"])

# ==== 5. Save result ====
df_all.to_csv("wwa_2014_2022_clean.csv", index=False)

print("Merging completed! Final dataset shape:", df_all.shape)
