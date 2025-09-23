#!/bin/bash

# ==============================
# Configurable variables
# ==============================
wfoOfInterest="HFO"         # <-- Change this to your WFO of interest
path_to="../../Weather_Data/"               # <-- Path containing {hfo}_wwa.csv
weather_data_folder="../../Weather_Data/HFO/"      # Folder where all weather data CSVs are saved

# ==============================
# Step 3: Clean the data
# ==============================
echo ">>> Step 3: Cleaning all weather data for $wfoOfInterest..."
python "Data Cleaning.py" "$weather_data_folder" "$wfoOfInterest"
if [ $? -ne 0 ]; then
    echo "Error in Data Cleaning.py"
    exit 1
fi

cleaned_folder="${weather_data_folder}/${wfoOfInterest}_cleaned"
echo "Cleaned files saved in: $cleaned_folder"

# ==============================
# Step 4: Create data points
# ==============================
wwa_file="${path_to}${wfoOfInterest}_wwa.csv"
training_file="${path_to}${wfoOfInterest}_Training_Data.csv"

echo ">>> Step 4: Creating training data points..."
python "CreateDataPoints.py" "$wwa_file" "$cleaned_folder" "$wfoOfInterest"
if [ $? -ne 0 ]; then
    echo "Error in CreateDataPoints.py"
    exit 1
fi
echo "Training data saved at: $training_file"

# ==============================
# Step 5: Normalise the data
# ==============================
clean_file="${path_to}${wfoOfInterest}_Training_Data.csv"
normalised_file="${path_to}${wfoOfInterest}_Training_Data_normalised.csv"

echo ">>> Step 5: Normalising training data..."
python "NormaliseData.py" "$clean_file"
if [ $? -ne 0 ]; then
    echo "Error in NormaliseData.py"
    exit 1
fi
echo "Normalised file saved at: $normalised_file"

echo ">>> Pipeline completed successfully!"
