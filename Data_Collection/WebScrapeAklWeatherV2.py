import requests
import json5
import pandas as pd
from bs4 import BeautifulSoup
import random
import time
from Weather_Data import CheckListForMissingData as CheckList

def webScrapADay(day,month,year, outputFile):
    url = f"https://www.timeanddate.com/scripts/cityajax.php?n=new-zealand/auckland&mode=historic&hd={year}{month}{day}&month={month}&year={year}&json=1"

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
    df.to_csv(f"Weather_Data/akl_weather_{day}_{month}_{year}.csv", index=False)
    outputFile.write("✅ CSV saved\n")
    print("✅ CSV saved")
    # print(df.head())

def butFirstGetMissedDates(missedDates, outputFile, forbiddenCount):

    print("Getting missed dates...")
    outputFile.write("Getting missed dates...\n")

    for strDay, strMonth, strYear in missedDates:
        print(f"{strDay}/{strMonth}/{strYear}", end= " ")
        outputFile.write(f"{strDay}/{strMonth}/{strYear} ")
        try:
            webScrapADay(strDay, strMonth, strYear, outputFile)
        except Exception as e:
            print(f"Failed to scrape {strDay}/{strMonth}/{strYear}: {e}")
            outputFile.write(f"Failed to scrape {strDay}/{strMonth}/{strYear}: {e}\n")
            if "403" in str(e):
                forbiddenCount += 1


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

def main():
    forbiddenCount = 0
    outputFile = open("output.log", "w")
    missedDates = CheckList.checkForMissingData()
    if len(missedDates) > 0:
        butFirstGetMissedDates(missedDates, outputFile, forbiddenCount)
    cycleThroughDates(outputFile, forbiddenCount)
    outputFile.close()

if __name__ == "__main__":
    main()