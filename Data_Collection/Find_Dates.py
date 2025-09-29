import sys
import pandas as pd

def main(wfoOfInterest, filePath):
    excludeTags = ["SU", "HY"]

    df = pd.read_csv(filePath)

    dfSOI = df[(df["wfo"] == wfoOfInterest) & (~df["phenom"].isin(excludeTags))]

    print(dfSOI)

    slashIndex = filePath.rfind("/")
    nexFilename = filePath[:slashIndex+1] + wfoOfInterest.upper() + "_wwa.csv"
    dfSOI.to_csv(nexFilename)

    print("Extracted :", wfoOfInterest, "| Saving to :", nexFilename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing inputs")
        sys.exit(1)

    filePath = sys.argv[1]

    wfoOfInterest = sys.argv[2].upper()
    main(wfoOfInterest, filePath)