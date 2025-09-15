#!/bin/bash

# Source directory (where the original + cleaned files are)
SOURCE_DIR="../../Weather_Data/HFO/"
# Target directory for cleaned files
CLEAN_DIR="${SOURCE_DIR}/clean/"

# Create the clean directory if it doesn't exist
mkdir -p "$CLEAN_DIR"

# Move all *_clean.* files into the clean folder
for file in "$SOURCE_DIR"/*_clean.*; do
    if [[ -f "$file" ]]; then
        echo "Moving $file -> $CLEAN_DIR"
        mv "$file" "$CLEAN_DIR"
    fi
done

echo "All cleaned files moved to $CLEAN_DIR"
