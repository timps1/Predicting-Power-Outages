import pandas as pd

'''
Keep outage_flag → as the most basic 0/1 label, suitable for our classification prediction, such as Random Forest
And, add aggregate features (outage_count, outage_sum) 
→ make the data more flexible, so maybe we don’t have to redo the cleaning if you need to change tasks later
'''

# Read
outages = pd.read_csv("eaglei_outages_2014_2022_clean.csv")
wwa = pd.read_csv("wwa_2014_2022_clean.csv")
wwa.columns = wwa.columns.str.lower()
outages.columns = outages.columns.str.lower()

# Data
outages["date"] = pd.to_datetime(outages["run_start_time"]).dt.date
wwa["date"] = pd.to_datetime(wwa["issued"]).dt.date

# Aggregate power outage data by state + date
outage_stats = (
    outages.groupby(["state", "date"])
    .agg(
        outage_count=("fips_code", "count"), # number of power outage records
        outage_sum=("sum", "sum") # the total of the sum field
    )
    .reset_index()
)

# Outage flag
outage_stats["outage_flag"] = 1

# Merge into weather data
merged = pd.merge(
    wwa,
    outage_stats,
    on=["state", "date"],
    how="left"
)

# Missing value filling (no power outage on that day)
merged["outage_flag"] = merged["outage_flag"].fillna(0).astype(int)
merged["outage_count"] = merged["outage_count"].fillna(0).astype(int)
merged["outage_sum"] = merged["outage_sum"].fillna(0).astype(int)

# Save
merged.to_csv("wwa_with_outages_detailed.csv", index=False)
print(merged.head(10))
