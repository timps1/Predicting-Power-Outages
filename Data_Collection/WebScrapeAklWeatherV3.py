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
            # print("Changing IP")
            # subprocess.run(["./reset_ip.sh"])
            # time.sleep(10)
            print("Failed to complete")
            sys.exit(1)
        


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
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/buffalo&mode=historic&hd=", "BUF"),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/riverton&mode=historic&hd=", 'RIW'),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/pocatello&mode=historic&hd=", "PIH"),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/phoenix&mode=historic&hd=", "PSR"),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/seattle&mode=historic&hd=", "SEW"),
        # ("https://www.timeanddate.com/scripts/cityajax.php?n=usa/oxnard&mode=historic&hd=", "LOX"),
        ("https://www.timeanddate.com/scripts/cityajax.php?n=@4993756&mode=historic&hd=", "APX"),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@4347823&mode=historic&hd=', 'LWX'), #added from sheets
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/anchorage&mode=historic&hd=', 'AFC'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@4543762&mode=historic&hd=', 'OUN'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@4791858&mode=historic&hd=', 'AKQ'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/tallahassee&mode=historic&hd=', 'TAE'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/baton-rouge&mode=historic&hd=', 'LIX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/juneau&mode=historic&hd=', 'AJK'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/mobile&mode=historic&hd=', 'MOB'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/fairbanks&mode=historic&hd=', 'AFG'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/jacksonville&mode=historic&hd=', 'JAX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/galveston&mode=historic&hd=', 'HGX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/boston&mode=historic&hd=', 'BOX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/tulsa&mode=historic&hd=', 'TSA'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/jackson&mode=historic&hd=', 'JAN'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@5142042&mode=historic&hd=', 'OKX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/chicago&mode=historic&hd=', 'LOT'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/midland&mode=historic&hd=', 'MAF'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/miami&mode=historic&hd=', 'MFL'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/charleston-sc&mode=historic&hd=', 'CHS'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/greenville-sc&mode=historic&hd=', 'GSP'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/memphis&mode=historic&hd=', 'MEG'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@7317906&mode=historic&hd=', 'LZK'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/lake-charles&mode=historic&hd=', 'LCH'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/springfield-mo&mode=historic&hd=', 'SGF'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@4480153&mode=historic&hd=', 'MHX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/shreveport&mode=historic&hd=', 'SHV'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/louisville&mode=historic&hd=', 'LMK'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/pueblo&mode=historic&hd=', 'PUB'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/wilmington-nc&mode=historic&hd=', 'ILM'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@4965671&mode=historic&hd=', 'GYX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/wichita&mode=historic&hd=', 'ICT'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/cleveland&mode=historic&hd=', 'CLE'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/corpus-christi&mode=historic&hd=', 'CRP'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/roanoke&mode=historic&hd=', 'RNK'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@4403710&mode=historic&hd=', 'EAX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/goodland&mode=historic&hd=', 'GLD'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/san-antonio&mode=historic&hd=', 'EWX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/medford&mode=historic&hd=', 'MFR'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/cheyenne&mode=historic&hd=', 'CYS'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/monterey&mode=historic&hd=', 'MTR'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/portland-or&mode=historic&hd=', 'PQR'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/duluth&mode=historic&hd=', 'DLH'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/st-louis&mode=historic&hd=', 'LSX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/denver&mode=historic&hd=', 'BOU'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/tampa&mode=historic&hd=', 'TBW'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/birmingham&mode=historic&hd=', 'BMX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/omaha&mode=historic&hd=', 'OAX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/amarillo&mode=historic&hd=', 'AMA'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/milwaukee&mode=historic&hd=', 'MKX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/el-paso&mode=historic&hd=', 'EPZ'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@5069802&mode=historic&hd=', 'GID'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/dodge-city&mode=historic&hd=', 'DDC'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/nashville&mode=historic&hd=', 'OHX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/topeka&mode=historic&hd=', 'TOP'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/raleigh&mode=historic&hd=', 'RAH'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/north-platte&mode=historic&hd=', 'LBF'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/brownsville&mode=historic&hd=', 'BRO'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/san-angelo&mode=historic&hd=', 'SJT'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/state-college&mode=historic&hd=', 'CTP'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/des-moines&mode=historic&hd=', 'DMX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/pittsburgh&mode=historic&hd=', 'PBZ'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/sioux-falls&mode=historic&hd=', 'FSD'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@4262285&mode=historic&hd=', 'IWX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/pontiac&mode=historic&hd=', 'DTX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/lubbock&mode=historic&hd=', 'LUB'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/tucson&mode=historic&hd=', 'TWC'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@4960140&mode=historic&hd=', 'CAR'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/grand-rapids&mode=historic&hd=', 'GRR'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/grand-forks&mode=historic&hd=', 'FGF'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/albany-ny&mode=historic&hd=', 'ALY'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/binghamton&mode=historic&hd=', 'BGM'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@5020881&mode=historic&hd=', 'MPX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/davenport&mode=historic&hd=', 'DVN'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/charleston-wv&mode=historic&hd=', 'RLX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/indianapolis&mode=historic&hd=', 'IND'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@7317167&mode=historic&hd=', 'ILX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/columbia&mode=historic&hd=', 'CAE'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/flagstaff&mode=historic&hd=', 'FGZ'),
        # ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/grand-junction&mode=historic&hd=', 'GJT'),
        # ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/la-crosse&mode=historic&hd=', 'ARX'),
        # ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/great-falls&mode=historic&hd=', 'TFX'),
        # ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/salt-lake-city&mode=historic&hd=', 'SLC'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/san-diego&mode=historic&hd=', 'SGX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/billings&mode=historic&hd=', 'BYZ'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/huntsville&mode=historic&hd=', 'HUN'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/missoula&mode=historic&hd=', 'MSO'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/spokane&mode=historic&hd=', 'OTX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/pendleton&mode=historic&hd=', 'PDT'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/reno&mode=historic&hd=', 'REV'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=@5392166&mode=historic&hd=', 'HNX'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/elko&mode=historic&hd=', 'LKN'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/sacramento&mode=historic&hd=', 'STO'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/boise&mode=historic&hd=', 'BOI'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/guam-hagatna&mode=historic&hd=', 'GUM'),
        ('https://www.timeanddate.com/scripts/cityajax.php?n=usa/las-vegas&mode=historic&hd=', 'VEF'),
        ]

    for prefixURL,wfoTag in infoList:
        try:
            Find_Dates.main(wfoTag, dataFilePath + "wwa_with_outages.csv")
            main(prefixURL, dataFilePath, wfoTag)
            print(f"Success {wfoTag}")
        except Exception as e:
            print(f"Failed {wfoTag}")
            print("ERROR:", e)