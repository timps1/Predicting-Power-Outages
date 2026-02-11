import os
import pandas as pd

# ==== Settings ====
dataset_folder = "dataset"  # folder containing all files
outages_file = os.path.join(dataset_folder, "wwa_with_outages.csv")

# Excluded phenom tags
exclude_tags = ["SU", "HY"]

# ==== Main Logic ====
def process_wfo(features_path, outages_df):
    # Extract WFO from filename, e.g. AJK_Training_Data_normalised.csv -> AJK
    filename = os.path.basename(features_path)
    wfo = filename.split("_")[0].upper()

    print(f"Processing {wfo}...")

    # Filter outage records for this WFO
    df_soi = outages_df[(outages_df["wfo"] == wfo) & (~outages_df["phenom"].isin(exclude_tags))]
    flags = df_soi["outage_flag"].reset_index(drop=True)

    # Load features
    features = pd.read_csv(features_path).reset_index(drop=True)

    # Align length
    n = min(len(features), len(flags))
    features = features.iloc[:n].copy()
    flags = flags.iloc[:n]

    # Add outage_flag
    features["outage_flag"] = flags

    # Create Weather_x columns
    for h in range(24):
        rain = f"RainValue_{h}"
        hail = f"HailValue_{h}"
        thunder = f"ThunderValue_{h}"
        weather = f"Weather_{h}"
        if all(col in features.columns for col in [rain, hail, thunder]):
            features[weather] = features[[rain, hail, thunder]].max(axis=1)

    # Drop Rain/Hail/Thunder
    drop_cols = [c for c in features.columns if "RainValue" in c or "HailValue" in c or "ThunderValue" in c]
    features = features.drop(columns=drop_cols)

    # Reorder columns like Hawaii
    new_cols = []
    for h in range(24):
        new_cols.extend([
            f"Temp_{h}",
            f"Weather_{h}",
            f"Wind Speed_{h}",
            f"Wind Direction_{h}",
            f"Humidity_{h}",
            f"Barometer_{h}"
        ])
    new_cols.append("outage_flag")
    features = features[new_cols]

    # Save output
    output_file = os.path.join(dataset_folder, f"{wfo}_Training_Data_with_outage.csv")
    features.to_csv(output_file, index=False)
    print(f"✅ {wfo} done. Saved to {output_file}")


def main():
    # Load outages big file once
    outages_df = pd.read_csv(outages_file)

    # Loop through all normalised files
    for file in os.listdir(dataset_folder):
        if file.endswith("_Training_Data_normalised.csv"):
            features_path = os.path.join(dataset_folder, file)
            process_wfo(features_path, outages_df)


if __name__ == "__main__":
    main()
