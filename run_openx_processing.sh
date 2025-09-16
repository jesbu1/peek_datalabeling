#!/bin/bash

# Configuration
SAVE_DIR=$1

if [ -z "$SAVE_DIR" ]; then
    echo "Error: SAVE_DIR is required as the first argument"
    echo "Usage: $0 <SAVE_DIR> [dataset_index_or_name]"
    echo "  SAVE_DIR: Directory to save processed data"
    echo "  dataset_index_or_name: Optional. Dataset index (number) or name. If not provided, processes all datasets."
    exit 1
fi

echo "Saving all processed data to $SAVE_DIR"

# List of datasets
DATASETS=(
    fractal20220817_data
    bridge_v2
    #kuka
    taco_play
    jaco_play
    #berkeley_cable_routing
    #roboturk doesn't work, don't use
    viola
    berkeley_autolab_ur5
    toto
    #language_table
    stanford_hydra_dataset_converted_externally_to_rlds
    austin_buds_dataset_converted_externally_to_rlds
    nyu_franka_play_dataset_converted_externally_to_rlds
    #furniture_bench_dataset_converted_externally_to_rlds
    ucsd_kitchen_dataset_converted_externally_to_rlds
    austin_sailor_dataset_converted_externally_to_rlds
    austin_sirius_dataset_converted_externally_to_rlds
    bc_z
    dlr_edan_shared_control_converted_externally_to_rlds
    iamlab_cmu_pickup_insert_converted_externally_to_rlds
    utaustin_mutex
    berkeley_fanuc_manipulation
    cmu_stretch
    #dobbe # doesn't work, don't use
    #fmb
    droid
)

# Determine which datasets to process
if [ $# -eq 2 ]; then
    # Dataset specified as second argument
    if [[ "$2" =~ ^[0-9]+$ ]]; then
        # Argument is a number, treat as index
        DATASET_INDEX=$2
        if [ $DATASET_INDEX -lt ${#DATASETS[@]} ]; then
            DATASET=${DATASETS[$DATASET_INDEX]}
            echo "Using dataset index $DATASET_INDEX: $DATASET"
            DATASETS_TO_PROCESS=("$DATASET")
        else
            echo "Error: Index $DATASET_INDEX is out of range. Max index is $((${#DATASETS[@]}-1))"
            exit 1
        fi
    else
        # Argument is a name
        DATASET="$2"
        echo "Using dataset name: $DATASET"
        DATASETS_TO_PROCESS=("$DATASET")
    fi
else
    # Process all datasets if only SAVE_DIR provided
    echo "No dataset specified, processing all ${#DATASETS[@]} datasets"
    DATASETS_TO_PROCESS=("${DATASETS[@]}")
fi

# Create log directory
mkdir -p ./log_files

# Main log file for all processing
MAIN_LOG_FILE="./log_files/oxe_processing_all_$(date +%Y%m%d_%H%M%S).log"

echo "$(date): Starting processing ${#DATASETS_TO_PROCESS[@]} datasets" | tee -a $MAIN_LOG_FILE

# Loop through each dataset
for DATASET in "${DATASETS_TO_PROCESS[@]}"; do
    # Skip if dataset is commented out
    if [[ "$DATASET" == \#* ]]; then
        echo "$(date): Dataset is commented out, skipping: ${DATASET#\#}" | tee -a $MAIN_LOG_FILE
        continue
    fi
    
    echo "$(date): Starting processing dataset: $DATASET" | tee -a $MAIN_LOG_FILE
    
    python label_videos_oxe.py --save_dir ${SAVE_DIR}/oxe_processed_subtrajectory --tfds_path openx_datasets --specific_tasks $DATASET 2>&1 | tee -a $MAIN_LOG_FILE
    
    EXIT_STATUS=$?
    
    # Log the exit status
    echo "$(date): Dataset $DATASET completed with exit status $EXIT_STATUS" | tee -a $MAIN_LOG_FILE
    
    # Optional: Add a short sleep between datasets to allow system to stabilize
    sleep 5
done

echo "$(date): All datasets processing completed." | tee -a $MAIN_LOG_FILE 
