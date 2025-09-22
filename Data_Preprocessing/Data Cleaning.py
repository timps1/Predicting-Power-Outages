import pandas as pd
import html
import re
import sys
import 

# 1) Load data
df = pd.read_csv(sys.argv[1])

# ---------- Helpers ----------
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
print("\nWeather code mapping (Code -> Weather_Text)")
print(mapping_df.to_string(index=False))

mapping_df.to_csv("weather_code_mapping.csv", index=False)

# Save cleaned dataset — 'Weather' is numeric-only
df.to_csv("weather_clean_with_codes.csv", index=False)
