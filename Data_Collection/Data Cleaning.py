import pandas as pd
import html
import re
import sys
import UniqueSetOfConditions as usc
import re
import os

def normalize_text(s):
    """Decode HTML (&nbsp;), normalize whitespaces, and trim."""
    if pd.isna(s):
        return s
    s = html.unescape(str(s))
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_number(text, pattern=r"(-?\d+(?:\.\d+)?)"):
    """Extract the first number using regex; return NA if not found."""
    if pd.isna(text):
        return pd.NA
    m = re.search(pattern, str(text))
    return float(m.group(1)) if m else pd.NA

def addRainValue(df, columnName):
    rainKeyWordsDict = {
        # Light precipitation
        "drizzle": 1,
        "sprinkles": 1,

        # Light–moderate
        "light rain": 2,
        "scattered showers": 2,
        "intermittent rain": 2,

        # Moderate
        "shower": 2.5,
        "rain showers": 3,
        "rain": 3,                 # steady rain

        # Strong storms
        "lots of rain": 4,
        "thundershowers": 4,
        "thunderstorm": 4,
        "heavy rain": 4.5,

        # Severe
        "severe thunderstorm": 5,
    }

    # Initialize column with 0
    df["RainValue"] = 0
    df["RainValue"] = df["RainValue"].astype(float)

    # Assign values based on keywords
    for phrase, score in rainKeyWordsDict.items():
        df.loc[
            df[columnName].astype(str).str.contains(phrase, case=False, na=False),
            "RainValue"
        ] = score

    return df

def addThunderValue(df, columnName):
    df["ThunderValue"] = 0
    for i, value in enumerate(df[columnName]):
        if isinstance(value, str) and "thunder" in value.lower():
            df.loc[i, "ThunderValue"] = 1
    return df

def addHailValue(df, columnName):
    df["HailValue"] = 0
    for i, value in enumerate(df[columnName]):
        if not isinstance(value, str):
            continue
        val = value.lower()
        if "hail" in val:
            df.loc[i, "HailValue"] = 1
        elif "sleet" in val:
            df.loc[i, "HailValue"] = 0.5
    return df

def addDiscriptiveInfo(df, columnName):
    addRainValue(df, columnName)
    addHailValue(df, columnName)
    addThunderValue(df, columnName)

def addWeatherColumn(df):
    def primary_weather(w):
        if pd.isna(w):
            return pd.NA
        w = normalize_text(w)
        return w.split(".", 1)[0].strip()

    # temporary text column
    df["_WeatherText"] = df["Weather"].apply(primary_weather)

    # stable alphabetical encoding: 1..K
    weather_vals = sorted(df["_WeatherText"].dropna().unique())
    weather2code = {w: i + 1 for i, w in enumerate(weather_vals)}

    # replace 'Weather' with numeric codes (nullable integer)
    df["Weather"] = df["_WeatherText"].map(weather2code).astype("Int64")

    # drop helper text column so only numbers remain
    df = df.drop(columns=["_WeatherText"])

    # Print and save mapping (Code -> Weather text) separately
    mapping_df = (
        pd.DataFrame({"Code": range(1, len(weather_vals) + 1), "Weather_Text": weather_vals})
        .astype({"Code": "int64"})
    )
    # print("\nWeather code mapping (Code -> Weather_Text)")
    # print(mapping_df.to_string(index=False))

    return df

def cleanOneFile(filename, filePath, nextFilePath):
    # 1) Load data
    df = pd.read_csv(filePath + filename)

    # ---------- Helpers ----------

    # 2) Normalize all object columns
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].apply(normalize_text)

    # 3) Temp -> numeric (°C)
    df["Temp"] = pd.to_numeric(df["Temp"].apply(lambda x: extract_number(x)), errors="coerce")

    # 4) Wind Speed -> numeric (km/h)
    df["Wind Speed"] = pd.to_numeric(df["Wind Speed"].apply(lambda x: extract_number(x)), errors="coerce")

    # 5) Wind Direction -> angle only (degrees)
    df["Wind Direction"] = pd.to_numeric(
        df["Wind Direction"].apply(lambda x: extract_number(x, pattern=r"(\d+(?:\.\d+)?)\s*°")),
        errors="coerce"
    )

    # 6) Drop Visibility if present
    if "Visibility" in df.columns:
        df = df.drop(columns=["Visibility"])

    # 7) Weather discriptions

    df = addWeatherColumn(df)

    addDiscriptiveInfo(df, "Weather")

    # df = df.drop(columns='Weather')

    df_temp_col_time = df["Time"]

    df = df.drop(columns='Time')

    df = df.replace(r'[^0-9\.-]', '', regex=True)

    df = df.apply(pd.to_numeric, errors='coerce')

    df = df.fillna(0.0)

    df["Humidity"] = df["Humidity"] / 100

    match = re.match(r"^\d{1,2}:\d{2}\s[ap]\.m\.", df_temp_col_time[0])
    if match:
        df_temp_col_time[0] = match.group(0)

    df["Time"] = df_temp_col_time

    df = addWeatherColumn(df)

    # Save cleaned dataset — 'Weather' is numeric-only
    dotIndex = filename.rfind(".")
    nexFilename = nextFilePath + "/" + filename[:dotIndex] + "_clean" + ".csv"
    df.to_csv(nexFilename, index=False)

    # print("File Saved to:", nexFilename)
    # print(df["Weather"])

def main():
    # Folder with saved data:
    folderWithFiles = sys.argv[1]
    if folderWithFiles[-1] != "/":
        folderWithFiles += "/"

    listOfFiles = os.listdir(folderWithFiles)

    wfo = sys.argv[2]

    cleanedFolder = folderWithFiles + f"/{wfo.upper()}_cleaned"

    if not os.path.isdir(cleanedFolder):
        os.mkdir(cleanedFolder)

    reboot = True
    
    for file in listOfFiles:
        if f"{wfo.lower()}_weather_" not in file:
            continue
        elif not os.path.exists(cleanedFolder + "/" + file[:file.rfind(".")] + "_clean" + ".csv") or reboot:
            try:
                cleanOneFile(file, folderWithFiles, cleanedFolder)
            except Exception as e:
                print(file)
                print("ERROR:", str(e))


if __name__ == "__main__":
    main()