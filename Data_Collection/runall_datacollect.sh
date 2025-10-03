#!/bin/bash

# ==============================
# Configurable variables
# ==============================
# ABQ ABR AFC AKQ APX BIS BTV BUF KEY HEB HGX FWD FFC GGW GRB HFO ILN JKL  LWX MLB MQT MRX OUN PAH PHI PIH PSR RIW SEW UNR
# HGX LOX
wfo_list=()   # <-- Add all WFOs of interest here
path_to="../../Weather_Data/"        # <-- Path containing {wfo}_wwa.csv

# ==============================
# Loop through each WFO
# ==============================
for wfoOfInterest in "${wfo_list[@]}"; do
    echo "========================================"
    echo ">>> Starting pipeline for $wfoOfInterest"
    echo "========================================"

    larger_dates_file="${path_to}wwa_with_outages.csv"
    weather_data_folder="${path_to}${wfoOfInterest}/"

    echo ">>> Step 1: Get dates for $wfoOfInterest..."
    python "Find_Dates.py" "$larger_dates_file" "$wfoOfInterest"

    # ==============================
    # Step 3: Clean the data
    # ==============================
    echo ">>> Step 3: Cleaning all weather data for $wfoOfInterest..."
    python "Data Cleaning.py" "$weather_data_folder" "$wfoOfInterest"
    if [ $? -ne 0 ]; then
        echo "Error in Data Cleaning.py for $wfoOfInterest"
        exit 1
    fi

    cleaned_folder="${weather_data_folder}/${wfoOfInterest}_cleaned"
    echo "Cleaned files saved in: $cleaned_folder"

    # ==============================
    # Step 4: Create data points
    # ==============================
    wwa_file="${path_to}${wfoOfInterest}_wwa.csv"
    training_file="${path_to}${wfoOfInterest}_Training_Data.csv"

    echo ">>> Step 4: Creating training data points for $wfoOfInterest..."
    python "CreateDataPoints.py" "$wwa_file" "$cleaned_folder" "$wfoOfInterest"
    if [ $? -ne 0 ]; then
        echo "Error in CreateDataPoints.py for $wfoOfInterest"
        exit 1
    fi
    echo "Training data saved at: $training_file"

    # ==============================
    # Step 5: Normalise the data
    # ==============================
    clean_file="${path_to}${wfoOfInterest}_Training_Data.csv"
    normalised_file="${path_to}${wfoOfInterest}_Training_Data_normalised.csv"

    echo ">>> Step 5: Normalising training data for $wfoOfInterest..."
    python "NormaliseData.py" "$clean_file"
    if [ $? -ne 0 ]; then
        echo "Error in NormaliseData.py for $wfoOfInterest"
        exit 1
    fi
    echo "Normalised file saved at: $normalised_file"

    echo ">>> Pipeline for $wfoOfInterest completed successfully!"
    echo
done

echo ">>> All WFO pipelines completed successfully!"
