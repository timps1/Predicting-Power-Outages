import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def normalize_time(t):
    return (t.replace("a.m.", "AM")
             .replace("p.m.", "PM"))

wwadfPath = "../../Weather_Data/Hawaii_wwa.csv"
wwadf = pd.read_csv(wwadfPath)

dataFilePath = "../../Weather_Data/HFO/clean/"
dataFilename = "hfo_weather_"

listOfFiles = os.listdir(dataFilePath)

stepFullDay = timedelta(days=1)
stepHalfDay = timedelta(hours=12)
stepHours = timedelta(hours=1)

dataList = []
first = False
second = False
third = False

for i in range(len(wwadf["issued"])):
    dataPoint = []

    try:
        issued_dt = datetime.strptime(wwadf['issued'][i].strip(), "%Y-%m-%d %H:%M:%S") - stepHalfDay

        filename_str = dataFilePath + dataFilename + issued_dt.strftime("%d_%m_%Y") + "_clean.csv"

        
        cleaneddf = pd.read_csv(filename_str)

        potentialStart  = issued_dt

        for j in range(len(cleaneddf['Time'])-1):

            # Parse entire Time column once into full datetimes
            first = True
            times = cleaneddf['Time'].apply(lambda t: normalize_time(t.strip()))
            first = False

            parsed_times = []
            current_date = issued_dt.date()
            prev_dt = None

            for t in times:
                # Always just take the time-of-day part
                parts = t.split()
                clock = parts[0] + " " + parts[1]   # e.g. "12:53 AM"
                parsed_time = datetime.strptime(clock, "%I:%M %p").time()
                dt = datetime.combine(current_date, parsed_time)

                # If the clock goes backwards (e.g. 11:53 PM → 12:53 AM), roll to next day
                if prev_dt and dt < prev_dt:
                    current_date = current_date + timedelta(days=1)
                    dt = datetime.combine(current_date, parsed_time)

                parsed_times.append(dt)
                prev_dt = dt

            parsed_times = np.array(parsed_times)

            # Find the closest time in the whole file
            diffs = np.abs(parsed_times - potentialStart)
            current_i = np.argmin(diffs)


        
        counter = 0
        while counter < 24:
            if current_i >= len(cleaneddf['Time']):
                issued_dt = potentialStart + stepFullDay
                filename_str = dataFilePath + dataFilename + issued_dt.strftime("%d_%m_%Y") + "_clean.csv"
                cleaneddf = pd.read_csv(filename_str)
                current_i = 0
            
            dataPoint += cleaneddf.iloc[current_i].to_list()

            counter += 1
            current_i += 1

        dataPoint.append(wwadf["outage_flag"][i])
        dataList.append(dataPoint)

        

    except Exception as e:
        if first:
            print(t)
        print(f"Error processing row {i}: {first} {second} {third} {wwadf['issued'][i].strip()} {e}")
        continue

print(len(dataList))
print(len(dataPoint))
print(len(cleaneddf.columns))
print(dataPoint)
base_cols = [c for c in cleaneddf.columns if c != "parsed_time"]
df = pd.DataFrame(
    dataList,
    columns=[f"{col}_{hour}" for hour in range(24) for col in base_cols] + ["outage_flag"]
)
print(df)
df.to_csv("Hawaii_Training_Data.csv", index=False)
    
    



    
