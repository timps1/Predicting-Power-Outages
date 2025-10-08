import numpy as np
import pandas as pd

import re

SEED = 114514
np.random.seed(SEED)

prefix = "AKL"
file_path = f"./{prefix}_Training_Data_normalised.csv"
processed_file = f"./{prefix}_processed.csv"

# Prefix to be processed for dimensionality reduction (statistics will be generated)
FEATURE_PREFIXES = [
    "Weather", "Wind Speed", "Wind Direction", "Humidity",
    "Barometer", "RainValue", "HailValue", "ThunderValue"
]

DROP_PREFIXES = ["Time"]

# Match column names such as Weather_0, Wind Speed_1, Humidity_12, etc.
def match_prefix_cols(columns, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}_\d+$")
    return [col for col in columns if pattern.match(col)]

def reduce_columns(df, prefix):
    cols = match_prefix_cols(df.columns, prefix)
    if not cols:
        return pd.DataFrame()  
    data = df[cols]
    return pd.DataFrame({
        f"{prefix}_max": data.max(axis=1),
        f"{prefix}_min": data.min(axis=1),
        f"{prefix}_std": data.std(axis=1),
        f"{prefix}_q1": data.quantile(0.25, axis=1),
        f"{prefix}_q3": data.quantile(0.75, axis=1),
        f"{prefix}_mean": data.mean(axis=1)
    })

if __name__ == "__main__":
    df = pd.read_csv(file_path)

    processed_parts = []

    for prefix in FEATURE_PREFIXES:
        reduced = reduce_columns(df, prefix)
        processed_parts.append(reduced)

    if "outage_flag" in df.columns:
        processed_parts.append(df[["outage_flag"]])
    else:
        raise ValueError("Column 'outage_flag' not found in the data.")

    final_df = pd.concat(processed_parts, axis=1)

    final_df.to_csv(processed_file, index=False)
    print(f"Processed data saved to {processed_file}")