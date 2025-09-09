import pandas as pd
import html
import re
import os
import numpy as np
import glob

# 1. Load data

path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(path, 'Data', 'Weather')

main_df = pd.DataFrame()

for file in glob.glob(os.path.join(data_path, 'wwa_*.csv')):
    print(f"Processing file: {file}")



    df = pd.read_csv(file,
                        usecols=["WFO",
                            "ISSUED",
                            "PHENOM",
                            "SIG",
                            "IS_EMERGENCY",
                            "WINDTAG"],
                        dtype={"WFO": str,
                            "ISSUED": str,
                            "PHENOM": str,
                            "SIG": str,
                            "IS_EMERGENCY": str,
                            "WINDTAG": str}
                    )

    main_df = pd.concat([main_df, df], ignore_index=True)

#Count No. of times WOFs appear

wfo_counts = main_df['WFO'].value_counts()
print("WFO Counts:")
print(wfo_counts)


