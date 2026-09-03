#!/bin/bash

# Get the directory where this run.sh script lives
basedir=$( cd "$(dirname "$0")" ; pwd -P )

# MAAP requires outputs to go to a specific output folder in the execution directory
mkdir -p output
outdir=${PWD}/output

# Change to the directory where your code lives
cd ${basedir}

# Force install dependencies into the active runtime environment
conda env update -f ${basedir}/requirements.yml

# Run the python script
# Execute python script using conda run to ensure it finds installed packages
conda run --no-capture-output -n base python DINOV2Segmentation.py \
    --hf_token "$HF_TOKEN" \
    --tif_url "$TIF_URL" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --threshold $THRESHOLD