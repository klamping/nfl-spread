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
  "2024-11-27"
  "2024-12-03"
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
    python3 tsfour/run-models.py -d "$date"
    # node ../combineResults.js -d "$date"
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
    node ./weightPredictions.js -d "$date" -l tsfour -w 4 
    node ./check-predictions.js -d "$date" -l tsfour
  done
  node ./combineAllResults.js
  node ./calcConfidencePercent.js
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
