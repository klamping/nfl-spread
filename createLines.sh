#!/bin/bash

dates=(
  "2024-09-25"
  "2024-10-02"
  "2024-10-09"
  "2024-10-16"
  "2024-10-23"
  "2024-10-30"
  "2024-11-06"
  "2024-11-13"
  "2024-11-20"
)

for i in "${!dates[@]}"; do
    date="${dates[$i]}"
    w=$((i + 4)) # Calculate w as index + 4
    # Run the command with the current date and calculated w
    node createCSV.js -d "$date" -w "$w"
done
