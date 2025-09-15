#!/bin/bash

# Input directory
INPUT_DIR="../../Weather_Data/HFO/"
# Output directory
OUTPUT_DIR="../../Weather_Data/HFO/cleaned/"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all files in the input directory
for file in "$INPUT_DIR"/*; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")

        # Skip files that already contain "_clean"
        if [[ "$filename" == *_clean.* ]]; then
            echo "Skipping already cleaned file: $file"
            continue
        fi

        echo "Processing $file ..."

        # Run your Python cleaning script
        python "Data Cleaning.py" "$file"

        # Build output filename with _clean before extension
        base="${filename%.*}"   # remove extension
        ext="${filename##*.}"   # get extension
        outfile="${OUTPUT_DIR}${base}_clean.${ext}"

        # Copy the cleaned output to the new file in output dir
        cp "weather_clean_with_codes.csv" "$outfile"

        echo "Saved cleaned file as $outfile"
    fi
done
