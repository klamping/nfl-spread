#!/bin/bash

# Define an array of dates
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

# Loop through each date using an index
for i in "${!dates[@]}"
do
  # Calculate the index-based value
  index_with_offset=$((i + 4))
  
  # Get the current date from the array
  date="${dates[i]}"
  
  # Run the command with the current date and calculated index
  node createCSV.js -d "$date" -w "$index_with_offset" -p
done
