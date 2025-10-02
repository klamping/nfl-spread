#!/bin/bash

models=(
    # 'lower'
    # 'middle'
    # 'middleUp'
    'upper'
    # 'one'
    # 'oneTwoThree'
    # 'twoThreeFour'
    # 'four'
    # 'fourFiveSix'
    'sevenEightNine'
    'aboveNine'
    # 'two'
    # 'three'
    # 'four'
    # 'fiveSix'
    # 'seven'
    # 'eight'
    # 'aboveEight'
)

for model in "${models[@]}"
do
    echo "---- Training model $model ----"
    # Run the command with the current date
    python3 train.py -m "$model"
done
