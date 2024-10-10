#!/bin/bash

# Default values
start_subject_idx=0 # index ranges from 0 to 39
subject_idx_step=1
fold='F1'
gpu=0
epoch=1000

# Check for command-line arguments and override defaults if provided
if [ "$#" -ge 3 ]; then
    start_subject_idx=$1
    subject_idx_step=$2
    fold=$3
fi

if [ "$#" -ge 4 ]; then
    gpu=$4
fi

if [ "$#" -ge 5 ]; then
    epoch=$5
fi

echo "Running with arguments:"
echo "  start_subject_idx: $start_subject_idx"
echo "  subject_idx_step: $subject_idx_step"
echo "  fold: $fold"
echo "  gpu: $gpu"
echo "  epoch: $epoch"

# Preprocess data
echo "Preprocessing sEMG data..."
bash preprocess_sEMG.sh "$start_subject_idx" "$subject_idx_step" "$fold"

# Evaluate noisy data
echo "Evaluating noisy test data..."
bash noisy.sh "$fold" "$gpu" "$epoch"

# Run NN-based denoising methods
echo "Starting NN-based denoising methods..."
bash train.sh "$fold" "$gpu" "$epoch"

# Run existing denoising methods
echo "Starting existing denoising methods..."
bash conventional.sh "$fold" "$gpu" "$epoch"