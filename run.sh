#!/bin/bash
# run.sh - Entrypoint for DPS system

# Exit immediately if a command exits with a non-zero status.
set -e

# Default environment variables (can be overridden by DPS runner)
HF_TOKEN=${HF_TOKEN:-""} 
TIF_URL=${TIF_URL:-"https://glihtdata.gsfc.nasa.gov/files/G-LiHT/AK_20180705_FIA_19/photography/orthomosaic/AK_20180705_FIA_19_l0s86_ortho.tif"}
MODEL_PATH=${MODEL_PATH:-"./model/model.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"output"}
THRESHOLD=${THRESHOLD:-0.30}

# Execute python script
python3 dps_processor.py \
    --hf_token "$HF_TOKEN" \
    --tif_url "$TIF_URL" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --threshold $THRESHOLD