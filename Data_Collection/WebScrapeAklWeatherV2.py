import random
import requests
import json5
import pandas as pd
from bs4 import BeautifulSoup
import time
import sys
from datetime import datetime, timedelta
import os
# from requests_ip_rotator import ApiGateway

def webScrapADay(day,month,year, dataFilePath, dataFilename, session):
    url = f"https://www.timeanddate.com/scripts/cityajax.php?n=usa/dallas&mode=historic&hd={year}{month}{day}&month={month}&year={year}&json=1"




    response = session.get(url)



    response.raise_for_status()
    data_json = json5.loads(response.text)  # Use json5 instead of response.json()

    # Columns we want
    columns = ["Time", "Temp", "Weather", "Wind Speed", "Wind Direction", "Humidity", "Barometer", "Visibility"]
    data = []

    for entry in data_json:
        c = entry["c"]
        
        # Extract Time
        time_html = c[0]["h"]
        time_soup = BeautifulSoup(time_html, "html.parser")
        time = time_soup.get_text(" ", strip=True)
        
        # Temp
        temp = c[2]["h"].replace("\u00a0", " ").strip()
        
        # Weather description
        weather = c[3]["h"].strip()
        
        # Wind speed
        wind_speed = c[4]["h"].strip()
        
        # Wind direction
        wind_html = c[5]["h"]
        wind_soup = BeautifulSoup(wind_html, "html.parser")
        wind_span = wind_soup.find("span")
        wind_direction = wind_span["title"] if wind_span and wind_span.has_attr("title") else ""
        
        # Humidity
        humidity = c[6]["h"].strip()
        
        # Barometer
        barometer = c[7]["h"].strip()
        
        # Visibility
        visibility = c[8]["h"].strip()
        
        data.append([time, temp, weather, wind_speed, wind_direction, humidity, barometer, visibility])

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f"{dataFilePath}{dataFilename}{day}_{month}_{year}.csv", index=False)
    print("✅ CSV saved")
    # print(df.head())

def butFirstGetMissedDates(missedDates, forbiddenCount, dataFilePath, dataFilename):

    with requests.Session() as session:
        for strDay, strMonth, strYear in missedDates:
            # time.sleep(random.random())
            print(f"{strDay}/{strMonth}/{strYear}", end= " ")
            try:
                webScrapADay(strDay, strMonth, strYear, dataFilePath, dataFilename, session)
            except Exception as e:
                print(f"Failed to scrape {strDay}/{strMonth}/{strYear}: {e}")
                if "403" in str(e):
                    forbiddenCount += 1
            if forbiddenCount > 10:
                print("Too many forbidden requests, stopping the script.")
                break

def organiseDates():
    
    df = pd.read_csv(sys.argv[1])
    
    hourStep = timedelta(hours=12)

    datesList = []

    for i in range(len(df['expired'])):
        issued_dt = datetime.strptime(df['issued'][i].strip(), "%Y-%m-%d %H:%M:%S")

        # adjust by buffer, then keep only the date
        start = (issued_dt - hourStep).date()
        end   = (issued_dt + hourStep).date()
        
        step = timedelta(days=1)
        current = start

        while current <= end:
            yearStr = current.strftime("%Y")
            monthStr = current.strftime("%m")
            dayStr = current.strftime("%d")
            datesList.append((dayStr, monthStr, yearStr)) 
            current += step
    
    if len(datesList) > 0:
        datesList = set(datesList)
        datesList = list(datesList)
    return datesList
        
def checkMissingDates(missedDates, dataFilePath, dataFilename):
    listOfFiles = os.listdir(dataFilePath)
    for i in range(len(missedDates)-1,-1,-1):
        searchDate = missedDates[i]
        dateStr = "_".join(searchDate) + ".csv"
        if dataFilename + dateStr in listOfFiles:
            missedDates.pop(i)


def main():
    if len(sys.argv) < 2:
        print("MISSING INPUT IN CMD LINE")
        sys.exit(1)
    dataFilePath = sys.argv[2]
    if not os.path.isdir(dataFilePath):
        os.mkdir(dataFilePath)
        print("Created Directory at", dataFilePath)
    if dataFilePath[-1] != "/":
        dataFilePath += "/"

    wfoTag = sys.argv[3]

    dataFilename = f"{wfoTag.lower()}_weather_"
    forbiddenCount = 0
    if len(sys.argv) < 2:
        print("MISSING INPUT IN CMD LINE")
        sys.exit(1)
    missedDates = organiseDates()
    print("Number of Dates:", len(missedDates))
    checkMissingDates(missedDates, dataFilePath, dataFilename)
    print("Number of Dates to go:", len(missedDates))
    if len(missedDates) > 0:
        print(missedDates[0])
        butFirstGetMissedDates(missedDates, forbiddenCount, dataFilePath, dataFilename)


if __name__ == "__main__":
    main()