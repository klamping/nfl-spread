#!/bin/bash

# Define an array of dates
dates=(
  "2025-09-23",
  "2025-09-30",
  "2025-10-07",
  "2025-10-14",
  "2025-10-21",
  "2025-10-28",
  "2025-11-04",
  "2025-11-11",
  "2025-11-18",
  "2025-11-25",
  "2025-12-02",
  "2025-12-09",
  "2025-12-16",
  "2025-12-23",
  "2025-12-30"
)

# Flag to determine whether to run the initial commands
run_initial_commands=false

# Parse CLI arguments
while getopts "i" opt; do
  case $opt in
    i)
      run_initial_commands=true
      ;;
    *)
      echo "Usage: $0 [-i]"
      exit 1
      ;;
  esac
done

# Run the initial commands if the flag is set
if [ "$run_initial_commands" = true ]; then
  # Loop through each date in the array
  for date in "${dates[@]}"
  do
    # Run the command with the current date
    python3 tsseven/run.py -d "$date"
  done
fi

# weights=(
#   '0.37'
#   '0.38'
#   '0.39'
#   '0.4'
#   '0.41'
# )
# for weight in "${weights[@]}"
# do
  echo "Predicting with weight $weight" 
  # Loop through each date in the array
  for date in "${dates[@]}"
  do
    # node ./weightPredictions.js -d "$date" -l tsseven
    node ./weightPDPredictions.js -d "$date" -l tsseven
    # node ./tsseven/combinePredictions.js -d "$date"
    node ./check-predictions.js -d "$date" -l tsseven -p
  done
  node ./combineAllResults.js -l tsseven
  # node ./calcConfidencePercent.js
# done

# Initialize cumulative values
# totalCorrectCount=0
# totalPredictions=0
# totalPoints=0

# # File to save JSON data
# output_file="results.json"

# # Clear the output file if it exists
# echo -n "" > "$output_file"

# # Write the initial JSON object to the output file
# echo '{"results": [' > "$output_file"

# # Loop through each date in the array
# for date in "${dates[@]}"
# do
#   result=$(node ./calc-results.js -d "$date" -l tsfour)
#   echo "$result"

#   # Save JSON data to the output file as a part of the results array
#   echo "$result," >> "$output_file"
# done

# # Remove the trailing comma and close the JSON object
# sed -i '$ s/,$//' "$output_file"
# echo -e "
# ]}" >> "$output_file"

# # Parse the JSON output to get cumulative values
# while IFS= read -r line; do
#   correctCount=$(echo "$line" | jq '.correctCount // 0')
#   predictions=$(echo "$line" | jq '.predictions // 0')
#   points=$(echo "$line" | jq '.totalPoints // 0')

#   # Add values to cumulative totals
#   totalCorrectCount=$((totalCorrectCount + correctCount))
#   totalPredictions=$((totalPredictions + predictions))
#   totalPoints=$((totalPoints + points))
# done < <(jq -c '.results[]' "$output_file")

# # Calculate the percentage of correct predictions with higher precision
# if [ "$totalPredictions" -gt 0 ]; then
#   correctPercentage=$(echo "scale=4; ($totalCorrectCount / $totalPredictions)" | bc)
# else
#   correctPercentage=0
# fi

# # Output the cumulative totals
# echo "Total Correct: $totalCorrectCount"
# echo "Total Points: $totalPoints"
# echo "Total Predictions: $totalPredictions"
# echo "Correct Percentage: $correctPercentage"
