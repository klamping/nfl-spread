#!/bin/bash

models=(
    'two'
    'three'
    'four'
    'fiveSix'
    'fourFiveSix'
    'seven'
    'eight'
    'aboveEight'
    'lower'
    'middle'
    'middleUp'
)

for model in "${models[@]}"
do
    echo "Training model $model"
    # Run the command with the current date
    python3 train.py -m "$model"
done
