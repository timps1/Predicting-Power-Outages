import pandas as pd

# Load the mapping table
mapping = pd.read_csv("../weather_code_mapping.csv")

# Find weather types just containing the 'storm' as a fragment
storm_codes = mapping[mapping["Weather_Text"].str.lower().str.contains("storm")]["Code"].tolist()
print("Storm-related weather codes:", storm_codes)

# Loading cleaned data
df = pd.read_csv("../weather_clean_with_codes.csv")

df_storm = df[df["Weather"].isin(storm_codes)]

print("\nStorm rows samples are:")
print(df_storm.head(20))

# Save the results (maybe not...)
df_storm.to_csv("weather_storm_filtered_method2.csv", index=False)
