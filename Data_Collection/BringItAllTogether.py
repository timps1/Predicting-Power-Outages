import os
import sys
import pandas as pd

if __name__ != "__main__":
    print("Don't call me (BRINGALLTOGETHER.py)")
    sys.exit(1)

folder = sys.argv[1]
output_file = "ALL_USA_TRAINING_DATA.csv"
header_written = False
counter = 0

for file in os.listdir(folder):
    if "_Training_Data_normalised.csv" in file and "NZ" not in file:
        file_path = os.path.join(folder, file)
        counter += 1
        # Read in chunks
        for chunk in pd.read_csv(file_path, chunksize=100000):  # Adjust chunk size if needed
            chunk.to_csv(output_file, mode='a', index=False, header=not header_written)
            header_written = True

print("Number of files added:", counter)
print(f"Merged CSV saved as {output_file}")
