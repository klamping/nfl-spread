#!/bin/bash

models=(
    'a'
    'oneTwoThree'
    'twoThreeFour'
    'four'
    'fourFiveSix'
    'sevenEightNine'
    'aboveNine'
)

for model in "${models[@]}"
do
    echo "Training model $model"
    # Run the command with the current date
    python3 train.py -m "$model"
done
