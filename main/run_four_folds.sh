#!/bin/bash

# Default values
start_subject_idx=0
subject_idx_step=1
gpu=0
epoch=1000

# Check for command-line arguments and override defaults if provided
if [ "$#" -ge 2 ]; then
    start_subject_idx=$1
    subject_idx_step=$2
fi

if [ "$#" -ge 3 ]; then
    gpu=$3
fi

if [ "$#" -ge 4 ]; then
    epoch=$4
fi

# Run the command for each fold with increasing start_subject_idx
for i in {0..3}; do
    start_subject_idx=$((i * 10))  # Increase by 10 for each iteration
    fold="F$((i + 1))"              # Generate fold names F1, F2, F3, F4
    echo "Running fold $fold with start_subject_idx $start_subject_idx..."
    bash run_one_fold.sh "$start_subject_idx" "$subject_idx_step" "$fold" "$gpu" "$epoch"
done
