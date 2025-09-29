import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])

a_list = []
print(df.columns)
for i, url in enumerate(df["URL"]):
    if url != 0 and url != "0":
        print(f"\t('{url}', '{df["Unique wfos"][i]}'),")
    
print(df)