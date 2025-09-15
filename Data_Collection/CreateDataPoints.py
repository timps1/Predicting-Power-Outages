import sys
import pandas as pd
import os
from datetime import datetime, timedelta

wwadfPath = "../../Hawaii_waa.csv"
wwadf = pd.read_csv(wwadfPath)

dataFilePath = "../../Weather_Data/HFO/"
dataFilename = "hfo_weather_"

listOfFiles = os.listdir(dataFilePath)
stepDay = timedelta(hours=24)
stepCoupleHours = timedelta(hours=4)

for i in range(len(wwadf["issued"])):
    issued_dt = datetime.strptime(wwadf['issued'][i].strip(), "%Y-%m-%d %H:%M:%S")
    expired_dt = datetime.strptime(wwadf['expired'][i].strip(), "%Y-%m-%d %H:%M:%S")

    start  = expired_dt - stepDay
    end = expired_dt

    if start > issued_dt - stepCoupleHours:
        start = issued_dt - stepCoupleHours
        end = start + stepDay
    
    



    
