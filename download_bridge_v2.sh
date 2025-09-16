#!/bin/bash
OPENX_DATASET_DIR=$1

if [ -z "$OPENX_DATASET_DIR" ]; then
    echo "Error: OPENX_DATASET_DIR is not set"
    exit 1
fi

echo "Downloading Bridge V2 to $OPENX_DATASET_DIR"

# Base URL of the TFDS dataset
BASE_URL="https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/"

# Create the output directory if it doesn't exist
mkdir -p "$OPENX_DATASET_DIR"

# Use wget to recursively download all files from the directory
wget --no-check-certificate -r -np -nH --cut-dirs=5 -R "index.html*" -P "$OPENX_DATASET_DIR" "$BASE_URL"

echo "Download completed. Files are saved in $OPENX_DATASET_DIR."

