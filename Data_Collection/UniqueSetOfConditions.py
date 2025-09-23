import pandas as pd
import sys

def conditionsSetConstruction(df, filename):
    print(df.columns)
    ConditionSet = df["Weather"].to_list()
    
    for i in range(len(ConditionSet)):
        if isinstance(ConditionSet[i], float):  # missing/NaN values
            ConditionSet[i] = "unknown"
        else:
            ConditionSet[i] = ConditionSet[i].lower()
    
    # get unique set
    ConditionSet = set(ConditionSet)
    print(ConditionSet)

    # create DataFrame with value column initialized to 0
    dfConditions = pd.DataFrame({
        "Conditions": list(ConditionSet)
    })

    # save to CSV
    dfConditions.to_csv(filename, index=False)

    return filename

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

    # Assign values based on keywords
    for phrase, score in rainKeyWordsDict.items():
        df.loc[df[columnName].str.contains(phrase, case=False, na=False), "RainValue"] = score

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


def main():
    df = pd.read_csv(sys.argv[1])
    filename = 'ConditionSet.csv'
    # filename = conditionsSetConstruction(df, filename)

    df = pd.read_csv(filename)
    addRainValue(df, "Conditions")
    addHailValue(df, "Conditions")
    addThunderValue(df, "Conditions")

    df.to_csv("ConditionSetValues.csv")


if __name__ == '__main__':
    main()