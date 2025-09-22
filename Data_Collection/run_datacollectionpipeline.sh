#!/bin/bash

# ==============================
# Configurable variables
# ==============================
wfoOfInterest="HFO"   # <-- Change this to your WFO of interest
path_to="../"         # <-- Change this to the path containing your *_wwa.csv
weather_data_folder="." # Default folder to save weather data (CURRENT DIR)

# ==============================
# Derived file paths
# ==============================
wwa_file="${path_to}${wfoOfInterest}_wwa.csv"
training_file="${path_to}${wfoOfInterest}_Training_Data.csv"
clean_file="${path_to}${wfoOfInterest}_Training_Data_clean.csv"
normalised_file="${path_to}${wfoOfInterest}_Training_Data_clean_normalised.csv"

# ==============================
# Step 3: Create data points
# ==============================
echo ">>> Step 3: Creating data points for $wfoOfInterest..."
python CreateDataPoints.py "$wwa_file" "$weather_data_folder" "$wfoOfInterest"
if [ $? -ne 0 ]; then
    echo "Error in CreateDataPoints.py"
    exit 1
fi
echo "Saved training file: $training_file"

# ==============================
# Step 4: Clean the data
# ==============================
echo ">>> Step 4: Cleaning data..."
python "Data Cleaning.py" "$training_file"
if [ $? -ne 0 ]; then
    echo "Error in Data Cleaning.py"
    exit 1
fi
echo "Saved cleaned file: $clean_file"

# ==============================
# Step 5: Normalise the data
# ==============================
echo ">>> Step 5: Normalising data..."
python NormaliseData.py "$clean_file"
if [ $? -ne 0 ]; then
    echo "Error in NormaliseData.py"
    exit 1
fi
echo "Saved normalised file: $normalised_file"

echo ">>> All steps completed successfully!"
