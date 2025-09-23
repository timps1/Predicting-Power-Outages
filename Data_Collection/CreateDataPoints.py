import sys
import pandas as pd
import os
from datetime import datetime, timedelta

wwadfPath = sys.argv[1]  # arg 1 = WWA file path
wwadf = pd.read_csv(wwadfPath)

dataFilePath = sys.argv[2]  # arg 2 = weather data dir (e.g. "../../HFO/")
if dataFilePath[-1] != "/":
    dataFilePath += "/"

wfoTag = sys.argv[3]
dataFilename = f"{wfoTag.lower()}_weather_"

stepDay = timedelta(hours=24)
halfDay = timedelta(hours=12)

results = []
columns_built = False
final_columns = []

for i in range(len(wwadf["issued"])):
    issued_dt = datetime.strptime(wwadf['issued'][i].strip(), "%Y-%m-%d %H:%M:%S")
    issued_time = issued_dt.time()

    start = issued_dt - halfDay
    end = issued_dt + halfDay
    start_date, end_date = start.date(), end.date()

    filenameAndPath = f"{dataFilePath}{dataFilename}{start.strftime('%d_%m_%Y')}_clean.csv"
    filenameAndPath2 = None
    if start_date != end_date:
        filenameAndPath2 = f"{dataFilePath}{dataFilename}{end.strftime('%d_%m_%Y')}_clean.csv"

    try:
        df = pd.read_csv(filenameAndPath)
        if filenameAndPath2:
            df2 = pd.read_csv(filenameAndPath2)
        else:
            df2 = None
    except Exception as e:
        print("Failed for weather:", i)
        print("ERROR:", e)
        continue

    # parse times
    df["ParsedTime"] = pd.to_datetime(
        df["Time"].str.strip().str.replace("a.m.", "AM", regex=False).str.replace("p.m.", "PM", regex=False),
        format="%I:%M %p"
    ).dt.time

    # find closest row
    idx = min(
        range(len(df["ParsedTime"])),
        key=lambda j: abs(
            datetime.combine(datetime.min, df["ParsedTime"][j]) -
            datetime.combine(datetime.min, issued_time)
        )
    )

    df = df.drop(columns = "ParsedTime")
    examplerow = df.iloc[idx]

    # collect 24 hours of rows starting at idx
    daySpanDataPoint = []
    counter = 0
    cur_df = df
    while counter < 24:
        row = cur_df.iloc[idx]
        daySpanDataPoint += list(row)

        idx += 1
        counter += 1
        if idx >= len(cur_df):
            if cur_df is df and df2 is not None:
                cur_df = df2
                idx = 0
            else:
                break

    # build column names once
    if not columns_built:
        base_cols = list(df.columns)
        for t in range(24):  # 0 through 24 hours
            for col in base_cols:
                final_columns.append(f"{col}_{t}")
        columns_built = True

    if len(daySpanDataPoint) == 24 * len(examplerow):
        results.append(daySpanDataPoint)

    length = len(wwadf["issued"])
    if i % max(length // 10, 1) == 0:  # print every 10% of progress
        print(f"{(i / length) * 100:.0f}%")

# build final DataFrame
training_df = pd.DataFrame(results, columns=final_columns)

cols_to_remove = ["Time_0", "Time_1", "Time_2", "Time_3"]  # your list here
for col in training_df.columns:
    if col.startswith("Time_") and col not in cols_to_remove:
        cols_to_remove.append(col)
training_df = training_df.drop(columns=cols_to_remove, errors="ignore")  # ignore avoids errors if not found

# save
slashIndex = sys.argv[1].rfind("/")
nextFilename = sys.argv[1][:slashIndex+1] + wfoTag.upper() + "_Training_Data.csv"
training_df.to_csv(nextFilename, index=False)

print(f"Saved training data to {nextFilename}")
