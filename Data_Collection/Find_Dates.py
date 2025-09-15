import sys
import pandas as pd

excludeTags = ["SU", "HY"]

print(sys.argv)

if len(sys.argv) < 3:
    print("Missing inputs")
    sys.exit(1)

print(sys.argv[1])

df = pd.read_csv(sys.argv[1])

stateOfInterest = "Hawaii"

dfSOI = df[(df["state"] == stateOfInterest) & (~df["phenom"].isin(excludeTags))]

print(dfSOI)

dfSOI.to_csv(sys.argv[2])
