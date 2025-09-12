import pandas as pd

# read
outages = pd.read_csv("eaglei_outages_2014_2022_clean.csv")
wwa = pd.read_csv("wwa_2014_2022_clean.csv")
wwa.columns = wwa.columns.str.lower()
outages.columns = outages.columns.str.lower()

# date format
outages["date"] = pd.to_datetime(outages["run_start_time"]).dt.date
wwa["date"] = pd.to_datetime(wwa["issued"]).dt.date

# In the power outage data, count whether there is a power outage by state + date
outage_flags = (
    outages.groupby(["state", "date"])
    .size()
    .reset_index(name="count")
)
outage_flags["outage_flag"] = 1   # outages occur → 1

# merge to weather data
merged = pd.merge(
    wwa,
    outage_flags[["state", "date", "outage_flag"]],
    on=["state", "date"],
    how="left"
)

# fill missing values: if there is no power outage recorded in the state on that day  → 0
merged["outage_flag"] = merged["outage_flag"].fillna(0).astype(int)

# save
merged.to_csv("wwa_with_outages.csv", index=False)

