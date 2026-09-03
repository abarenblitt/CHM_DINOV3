#!/bin/bash
set -e

# 1. Catch the positional arguments from MAAP
HF_TOKEN_ARG=$1
TIF_URL_ARG=$2
THRESHOLD_ARG=$3

# 2. Set up directories
basedir=$( cd "$(dirname "$0")" ; pwd -P )
mkdir -p output
export OUTPUT_DIR=${PWD}/output
cd ${basedir}

# 3. Install packages
echo "Updating Conda environment..."
conda env update -f ${basedir}/requirements.yml

# 4. Run the script with the caught variables
echo "Running Python script..."
conda run --no-capture-output -n base python DINOV2Segmentation.py \
    --hf_token "${HF_TOKEN_ARG}" \
    --tif_url "${TIF_URL_ARG}" \
    --model_path "./model/model.pt" \
    --output_dir "${OUTPUT_DIR}" \
    --threshold "${THRESHOLD_ARG}"