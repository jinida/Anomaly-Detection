#!/bin/bash

# Define the root path for the MVTec dataset
ROOT_PATH="/datasets/MVTec"

# List of categories to process
CATEGORIES=(
    "bottle"
    "cable"
    "capsule"
    "hazelnut"
    "leather"
    "screw"
    "tile"
)

# Loop through each category and run the training command
for CATEGORY in "${CATEGORIES[@]}"; do
    echo "--- Training for category: $CATEGORY ---"
    python train.py --root_path "$ROOT_PATH" --category "$CATEGORY"
    echo "--- Finished training for category: $CATEGORY ---"
    echo "" # Add a blank line for better readability between runs
done

echo "All training processes completed."