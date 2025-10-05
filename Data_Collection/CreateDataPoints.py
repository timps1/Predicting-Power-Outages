import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def createUSAPoints():
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
        try:
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
                print("Likely missing weather data")
                print("ERROR:", str(e)[:100], ".....")
                continue

            # parse times
            try:
                # df["ParsedTime"] = pd.to_datetime( df["Time"].str.strip().str.replace("a.m.", "AM", regex=False).str.replace("p.m.", "PM", regex=False), format="%I:%M %p" ).dt.time
                df["ParsedTime"] = (
                    df["Time"]
                    .str.strip()
                    .str.replace("a.m.", "AM", regex=False)
                    .str.replace("p.m.", "PM", regex=False)
                    .str.extract(r"(\d{1,2}:\d{2}\s(?:AM|PM|am|pm))")[0]  # extract just the time part
                    .str.upper()  # normalize am/pm to AM/PM
                    .pipe(pd.to_datetime, format="%I:%M %p", errors="coerce")
                    .dt.time
                )
            except: 
                print(df.loc[df["ParsedTime"].isna(), "Time"])
                print(df["Time"].str.strip())
                sys.exit(1)

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

            if len(daySpanDataPoint) == 24 * len(examplerow):
                if not columns_built:
                    base_cols = list(df.columns)
                    for t in range(24):  # 0 through 24 hours
                        for col in base_cols:
                            final_columns.append(f"{col}_{t}")
                    final_columns.append("date")
                    final_columns.append('outage_flag')
                    columns_built = True
                daySpanDataPoint += [issued_dt]
                daySpanDataPoint += [wwadf['outage_flag'][i]]

                results.append(daySpanDataPoint)
            else:
                print(24 * len(examplerow), len(daySpanDataPoint))

            length = len(wwadf["issued"])
            if i % max(length // 10, 1) == 0:  # print every 10% of progress
                print(f"{(i / length) * 100:.0f}%")
        
        except Exception as e:
            print(filenameAndPath)
            print("ERROR:", e)
            


        

    # build final DataFrame
    print("Size of DF:", len(results))
    training_df = pd.DataFrame(results, columns=final_columns)

    # cols_to_remove = ["Time_0", "Time_1", "Time_2", "Time_3"]  # your list here
    # for col in training_df.columns:
    #     if col.startswith("Time_") and col not in cols_to_remove:
    #         cols_to_remove.append(col)
    # training_df = training_df.drop(columns=cols_to_remove, errors="ignore")  # ignore avoids errors if not found

    # save
    slashIndex = sys.argv[1].rfind("/")
    nextFilename = sys.argv[1][:slashIndex+1] + wfoTag.upper() + "_Training_Data.csv"
    training_df.to_csv(nextFilename, index=False)

    print(f"Saved training data to {nextFilename}")


def createNZpoint1():

    tag = sys.argv[3]

    print(f"Creating {tag} data points using method one (day method)")
    NZFolder = sys.argv[2]
    print(f"Using folder with weather: {NZFolder}")

    listOfFiles = os.listdir(NZFolder)
    print(f"Number of Files {len(listOfFiles)}")

    results = []
    final_columns = []
    debugging = 0
    debuggingCounter = 0
    for file in listOfFiles:
        
        #Checking searching correct file
        if len(file) < 4:
            continue
        elif ".csv" not in file[-4:]:
            continue
        
        #getting date
        try:
            date_str = file.replace(f"{tag.lower()}_weather_", "").replace("_clean.csv", "")
            fileDate = datetime.strptime(date_str, "%d_%m_%Y")
        except Exception as e:
            print(f"Skipping {file}, bad date parse: {e}")
            continue

        df = pd.read_csv(os.path.join(NZFolder, file))
        numberOfRows = len(df[df.columns[0]])

        if numberOfRows < 24: # insifficient data
            continue
        
        # Extract HH:MM (ignore AM/PM)
        df["ParsedTime"] = df["Time"].str.strip().str.extract(r"(\d{1,2}:\d{2})")[0]

        # Convert to datetime.time (24-hour)
        df["ParsedTime"] = pd.to_datetime(df["ParsedTime"], format="%H:%M", errors="coerce").dt.time

        # Drop invalid rows
        df = df.dropna(subset=["ParsedTime"])

        df["ParsedDateTime"] = df.apply(
            lambda row: datetime.combine(fileDate, row["ParsedTime"]),
            axis=1
        )

        df = df.drop(columns="ParsedTime")

        counter = 0
        previousRow = None
        idx = 0
        tolTime = timedelta(minutes=5)
        oneHourStep = timedelta(hours=1)
        halfHourStep = timedelta(minutes=30)
        currentRow = df.iloc[idx]
        
        daySpanDataPoint = []
        while counter < 24:
            counter += 1
            daySpanDataPoint += list(currentRow.drop("ParsedDateTime")) #adding current row
            
            #determining next row
            previousRow = currentRow
            target_time = previousRow["ParsedDateTime"] + oneHourStep
            previousCalculation = abs(currentRow["ParsedDateTime"] - target_time)

            while idx + 1 < numberOfRows:
                candidate = df.iloc[idx + 1]
                candidate_diff = abs(candidate["ParsedDateTime"] - target_time)

                if candidate_diff < previousCalculation:
                    idx += 1
                    currentRow = candidate
                    previousCalculation = candidate_diff
                else:
                    break

        df = df.drop(columns="ParsedDateTime")
        # if debugging != len(daySpanDataPoint):
        #     print(len(daySpanDataPoint), date_str)
        #     print(daySpanDataPoint)
        #     debugging = len(daySpanDataPoint)
        #     debuggingCounter +=1
        #     if debuggingCounter == 4:
        #         sys.exit(1)
        if len(daySpanDataPoint) == 240:
            if len(final_columns) == 0:
                base_cols = list(df.columns)
                for t in range(24):  # 0 through 24 hours
                    for col in base_cols:
                        final_columns.append(f"{col}_{t}")
                final_columns.append("date")
                final_columns.append('outage_flag')
                print("Final columns:", final_columns)
            daySpanDataPoint += [fileDate, 0]
            results.append(daySpanDataPoint)

    # build final DataFrame
    print("Size of DF:", len(results))
    if len(results) != 0:
        training_df = pd.DataFrame(results, columns=final_columns)
        training_df = addKiwiFlags(training_df, tag) 
        nextFilename = f"{tag}_Training_Data.csv"
        training_df.to_csv(nextFilename, index=False)

        print(f"Saved training data to {nextFilename}")

def addKiwiFlags(df, tag, fileName = None):
    
    if fileName is not None:
        df = pd.read_csv(fileName)

    place_dict = {
        "AKL" : "North North Island",
        "WLGNZ" : "South North Island",
        "CHCNZ" : "South Island",
        }
    outageDF = pd.read_excel(sys.argv[1])
    outageDFForRegion = outageDF[(outageDF["SITE_REGION"] == place_dict[tag])]
    print(f"Number of outages: {len(outageDFForRegion)}")
    counter = 0
    for i, date in enumerate(outageDFForRegion["INTERRUPTION_DATETIME"]):
        date = str(date)
        dateFormated = date[:date.find(" ")]
        df.loc[df["date"] == dateFormated, "outage_flag"] = 1
        if (df["date"] == dateFormated).any():
            counter += 1
    
    print("Number of outages added to training data:", counter)
    
    return df

        

    



if __name__ == "__main__":
    if "NZ" not in sys.argv[3] and sys.argv[3] != "AKL":
        createUSAPoints()
    else:
        # addKiwiFlags(None, sys.argv[3], "CHCNZ_Training_Data.csv")
        createNZpoint1()