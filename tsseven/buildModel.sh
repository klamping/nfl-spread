#!/bin/bash

# Shell script to run a Python3 command 30 times

# Define the Python command
PYTHON_COMMAND="python3 train.py -m aboveEight"

# Loop to execute the command 30 times
for i in {1..30}
do
  echo "Execution $i:"
  $PYTHON_COMMAND
  if [ $? -ne 0 ]; then
    echo "Command failed on iteration $i. Exiting."
    exit 1
  fi
done

echo "Command executed successfully 30 times."
