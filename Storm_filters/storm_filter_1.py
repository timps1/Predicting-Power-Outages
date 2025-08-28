"""
This script implements two methods for detecting storms in the weather dataset:

1. Realistic indicator-based storm detection: (storm_filter_1)
    1)Uses physical measurements such as wind speed and barometric pressure.
    2)Flags storms if wind speed exceeds a threshold (e.g., >60 km/h),
    or if barometric pressure drops sharply (e.g., >5 hPa within one hour).
    3)Advantage: Captures meteorological conditions consistent with storms.
    4)Limitation: May include false positives (e.g., strong wind but no storm label).

2. Keyword-based storm detection: (storm_filter_2)
    1)Uses the textual weather description (Weather_Text).
    2)Selects all rows where the label contains the word 'storm'.
    3)Advantage: Directly captures explicit mentions of storms.
    4)Limitation: Misses cases where no keyword is present, even if conditions are storm-like.

Both methods complement each other:
The keyword method ensures alignment with reported weather labels.
The realistic method ensures coverage of meteorological extremes.
"""

import pandas as pd

# Read the file (cleaned data...)
df = pd.read_csv("../weather_clean_with_codes.csv")

# If Humidity has %, remove it and convert to float
if 'Humidity' in df.columns:
    df['Humidity'] = df['Humidity'].astype(str).str.rstrip('%')
    df['Humidity'] = pd.to_numeric(df['Humidity'], errors='coerce')

# Cleaning up the Barometer columns
if 'Barometer' in df.columns:
    df['Barometer'] = df['Barometer'].astype(str).str.replace(' mbar', '', regex=False)
    df['Barometer'] = pd.to_numeric(df['Barometer'], errors='coerce')

# Same as Wind Speed
if 'Wind Speed' in df.columns:
    df['Wind Speed'] = df['Wind Speed'].astype(str).str.replace(' km/h', '', regex=False)
    df['Wind Speed'] = pd.to_numeric(df['Wind Speed'], errors='coerce')

# Storm detection based on some real indicators (we can make changes later)
def detect_storm_realistic(row, df=df, wind_thresh=60, pressure_drop_thresh=5):
    # like wind speed is higher than 60...
    wind_flag = pd.notnull(row.get('Wind Speed')) and row['Wind Speed'] > wind_thresh

    pressure_flag = False
    if pd.notnull(row.get('Barometer')):
        idx = row.name
        if idx > 0 and pd.notnull(df.loc[idx-1, 'Barometer']):
            drop = df.loc[idx-1, 'Barometer'] - row['Barometer']
            pressure_flag = drop > pressure_drop_thresh

    return int(wind_flag or pressure_flag)

df['Storm_Flag'] = df.apply(detect_storm_realistic, axis=1)

df_storm_realistic = df[df['Storm_Flag'] == 1]

# Save the results (maybe not...)
df_storm_realistic.to_csv("weather_storm_filtered_method1.csv", index=False)

print(df_storm_realistic.head(20))
