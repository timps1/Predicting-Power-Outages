import sys
import pandas as pd

excludeTags = ["SU", "HY"]

if len(sys.argv) < 2:
    print("Missing inputs")
    sys.exit(1)

print(sys.argv[1])

df = pd.read_csv(sys.argv[1])

wfoOfInterest = "HFO"

dfSOI = df[(df["wfo"] == wfoOfInterest) & (~df["phenom"].isin(excludeTags))]

print(dfSOI)

slashIndex = sys.argv[1].rfind("/")
nexFilename = sys.argv[1][:slashIndex+1] + wfoOfInterest.upper() + "_wwa.csv"
dfSOI.to_csv(nexFilename)
