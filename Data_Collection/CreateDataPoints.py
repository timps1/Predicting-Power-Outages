import sys
import pandas as pd
import os
from datetime import datetime, timedelta

wwadfPath = sys.argv[1] # arg 1 needs to be where the hawaii data wwa file is
wwadf = pd.read_csv(wwadfPath)

dataFilePath = sys.argv[2] # arg 2 needs to be where the hawaii data is "../../HFO/"
if dataFilePath[-1] != "/":
    dataFilePath += "/"

wfoTag = sys.argv[3]
dataFilename = f"{wfoTag.lower()}_weather_"

listOfFiles = os.listdir(dataFilePath)
stepDay = timedelta(hours=24)
halfDay = timedelta(hours=12)
stepCoupleHours = timedelta(hours=4)

for i in range(len(wwadf["issued"])):
    issued_dt = datetime.strptime(wwadf['issued'][i].strip(), "%Y-%m-%d %H:%M:%S")

    start  = issued_dt - halfDay
    end = issued_dt + halfDay

    ### Closest hour code ###
    ### Flatten to 24 hour span ###


### Create DF ###

### Save DF ###
slashIndex = sys.argv[1].rfind("/")
nexFilename = sys.argv[1][:slashIndex+1] + wfoTag.upper() + "_Training_Data.csv"
# df.to_csv(nexFilename, index=False)

    
    



    
