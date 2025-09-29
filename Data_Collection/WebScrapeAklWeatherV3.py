import requests
import json5
import pandas as pd
from bs4 import BeautifulSoup
import time
import sys
from datetime import datetime, timedelta
import os
import subprocess
import Find_Dates

def webScrapADay(day,month,year, dataFilePath, dataFilename, prefixURL):
    url = f"{prefixURL}{year}{month}{day}&month={month}&year={year}&json=1"

    response = requests.get(url)
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
    print(f"✅ CSV saved")
    # print(df.head())

def butFirstGetMissedDates(missedDates, forbiddenCount, dataFilePath, dataFilename, prefixURL):
    counter = 0
    lengthOfMissed = len(missedDates)
    for strDay, strMonth, strYear in missedDates:
        print(f"{counter}/{lengthOfMissed} -> {strDay}/{strMonth}/{strYear}", end= " ")
        counter += 1
        try:
            webScrapADay(strDay, strMonth, strYear, dataFilePath, dataFilename, prefixURL)
        except Exception as e:
            print(f"Failed to scrape {strDay}/{strMonth}/{strYear}: {e}")
            if "403" in str(e):
                forbiddenCount += 1
        if forbiddenCount > 10:
            print("Too many forbidden requests, stopping the script.")
            break
    return forbiddenCount

def organiseDates(filename):
    
    print("Pulling dates from:", filename)

    df = pd.read_csv(filename)
    
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

    print("Checking dates at:", dataFilePath)

    listOfFiles = os.listdir(dataFilePath)
    for i in range(len(missedDates)-1,-1,-1):
        searchDate = missedDates[i]
        dateStr = "_".join(searchDate) + ".csv"
        if dataFilename + dateStr in listOfFiles:
            missedDates.pop(i)


def main(prefixURL, dataFilePath, wfoTag):

    copydataFilePath = dataFilePath
    dataFilePath = dataFilePath + wfoTag.upper()
    if not os.path.isdir(dataFilePath):
        os.mkdir(dataFilePath)
        print("Created Directory at", dataFilePath)
    if dataFilePath[-1] != "/":
        dataFilePath += "/"

    dataFilename = f"{wfoTag.lower()}_weather_"
    forbiddenCount = 0
    
    firstRun = True
    fails = 0
    while (forbiddenCount > 0 or firstRun) and fails < 3:
        forbiddenCount = 0
        firstRun = False
        missedDates = organiseDates(copydataFilePath + f"{wfoTag.upper()}_wwa.csv")
        print("Number of Dates:", len(missedDates))
        checkMissingDates(missedDates, dataFilePath, dataFilename)
        print("Number of Dates to go:", len(missedDates))
        if len(missedDates) > 0:
            print(missedDates[0])
            forbiddenCount = butFirstGetMissedDates(missedDates, forbiddenCount, dataFilePath, dataFilename, prefixURL)

        fails +=1
        if forbiddenCount > 0:
            print("Changing IP")
            subprocess.run(["./reset_ip.sh"])
            time.sleep(10)
        


if __name__ == "__main__":
    
    dataFilePath = "../../Weather_Data/"
    infoList = [
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/albuquerque&mode=historic&hd=", 'ABQ'),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/green-bay&mode=historic&hd=", 'GRB'),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/bismarck&mode=historic&hd=", 'BIS'),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=@4503134&mode=historic&hd=", 'PHI'),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/paducah&mode=historic&hd=", 'PAH'),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/marquette&mode=historic&hd=", 'MQT'),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=@5654320&mode=historic&hd=", "GGW"),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/burlington-vt&mode=historic&hd=", "BTV"),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/knoxville&mode=historic&hd=", "MRX"),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/melbourne&mode=historic&hd=", "MLB"),
        ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/buffalo&mode=historic&hd=", "BUF"),
        ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/riverton&mode=historic&hd=", 'RIW'),
        ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/pocatello&mode=historic&hd=", "PIH"),
        ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/phoenix&mode=historic&hd=", "PSR"),
        ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/seattle&mode=historic&hd=", "SEW"),
        ("https://www.timeanddate.com/scripts/cityajax.php?n=@4993756&mode=historic&hd=", "APX")
        ]

    for prefixURL,wfoTag in infoList:
        try:
            Find_Dates.main(wfoTag, dataFilePath + "wwa_with_outages.csv")
            main(prefixURL, dataFilePath, wfoTag)
            print(f"Success {wfoTag}")
        except Exception as e:
            print(f"Failed {wfoTag}")
            print("ERROR:", e)