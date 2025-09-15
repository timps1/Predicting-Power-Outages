import requests
import json5
import pandas as pd
from bs4 import BeautifulSoup
import random
import time
import sys
from datetime import datetime, timedelta
import os

def webScrapADay(day,month,year, outputFile, dataFilePath, dataFilename):
    url = f"https://www.timeanddate.com/scripts/cityajax.php?n=usa/honolulu&mode=historic&hd={year}{month}{day}&month={month}&year={year}&json=1"

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
    outputFile.write("✅ CSV saved\n")
    print("✅ CSV saved")
    # print(df.head())

def butFirstGetMissedDates(missedDates, outputFile, forbiddenCount, dataFilePath, dataFilename):

    for strDay, strMonth, strYear in missedDates:
        print(f"{strDay}/{strMonth}/{strYear}", end= " ")
        outputFile.write(f"{strDay}/{strMonth}/{strYear} ")
        try:
            webScrapADay(strDay, strMonth, strYear, outputFile, dataFilePath, dataFilename)
        except Exception as e:
            print(f"Failed to scrape {strDay}/{strMonth}/{strYear}: {e}")
            outputFile.write(f"Failed to scrape {strDay}/{strMonth}/{strYear}: {e}\n")
            if "403" in str(e):
                forbiddenCount += 1
        if forbiddenCount > 10:
            print("Too many forbidden requests, stopping the script.")
            break


def cycleThroughDates(outputFile, forbiddenCount):

    startYear, endYear = 2023, 2025
    startMonth, endMonth = 1, 12
    startDay, endDay = 1, 31
    for year in range(startYear, endYear+1):

        strYear = str(year)

        for month in range(startMonth, endMonth+1):

            strMonth = str(month)
            if month < 10:
                strMonth = "0" + strMonth
            
            for day in range(startDay,endDay+1):
                if month == 2 and day > 28:
                    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):  # Leap year check
                        if day > 29:
                            continue
                    else:
                        continue
                elif month in [4, 6, 9, 11] and day > 30:
                    continue

                strDay = str(day)
                if day < 10:
                    strDay = "0" + strDay
                print(f"{strDay}/{strMonth}/{strYear}", end= " ")
                outputFile.write(f"{strDay}/{strMonth}/{strYear} ")
                try:
                    webScrapADay(strDay,strMonth,strYear, outputFile)
                except Exception as e:
                    print("FAILED", e)
                    outputFile.write(f"Failed {e}\n")
                    if "403" in str(e):
                        forbiddenCount += 1
                
                number = random.randint(1,3)

                # time.sleep(number) #website was getting suspcious
            
            startDay = 1  # Reset day to 1 after the first month
            if forbiddenCount > 10:
                break
        
        startMonth = 1  # Reset month to January after the first year
        if forbiddenCount > 10:
            print("Too many forbidden requests, stopping the script.")
            outputFile.write("Too many forbidden requests, stopping the script.\n")
            break

def organiseDates():

    if len(sys.argv) < 2:
        print("MISSING INPUT IN CMD LINE")
        sys.exit(1)
    
    df = pd.read_csv(sys.argv[1])
    
    hourStep = timedelta(hours=8)

    datesList = []

    for i in range(len(df['expired'])):
        issued_dt = datetime.strptime(df['issued'][i].strip(), "%Y-%m-%d %H:%M:%S")
        expired_dt = datetime.strptime(df['expired'][i].strip(), "%Y-%m-%d %H:%M:%S")

        # adjust by buffer, then keep only the date
        start = (issued_dt - hourStep).date()
        end   = (expired_dt + hourStep).date()
        
        step = timedelta(days=1)
        current = start

        while current <= end:
            yearStr = current.strftime("%Y")
            monthStr = current.strftime("%m")
            dayStr = current.strftime("%d")
            datesList.append([dayStr, monthStr, yearStr]) 
            current += step
    datesList = set(datesList)
    datesList = list(datesList)
    return datesList
        
def checkMissingDates(missedDates, dataFilePath, dataFilename):
    listOfFiles = os.listdir(dataFilePath)
    happened = False
    for i in range(len(missedDates)-1,-1,-1):
        searchDate = missedDates[i]
        dateStr = "_".join(searchDate) + ".csv"
        if not happened:
                print(searchDate)
                print(dataFilename + dateStr)
                print(listOfFiles[0])
                happened = True
        if dataFilename + dateStr in listOfFiles:
            if not happened:
                print("Works!!!!!")
            missedDates.pop(i)


def main():
    dataFilePath = "../../Weather_Data/HFO/"
    dataFilename = "hfo_weather_"
    forbiddenCount = 0
    outputFile = open("output.log", "w")
    if len(sys.argv) < 2:
        print("MISSING INPUT IN CMD LINE")
        sys.exit(1)
    missedDates = organiseDates()
    checkMissingDates(missedDates, dataFilePath, dataFilename)
    print("Number of dates:", len(missedDates))
    print(missedDates[0])
    if len(missedDates) > 0:
        butFirstGetMissedDates(missedDates, outputFile, forbiddenCount, dataFilePath, dataFilename)
    outputFile.close()

if __name__ == "__main__":
    main()