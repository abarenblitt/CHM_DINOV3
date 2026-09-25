#!/bin/bash
set -e

# 1. Catch positional arguments automatically sent by MAAP
HF_TOKEN=$1
TIF_URL=$2
THRESHOLD=$3

# 2. Set directories
basedir=$( cd "$(dirname "$0")" ; pwd -P )
mkdir -p output
export OUTPUT_DIR=${PWD}/output
cd ${basedir}

# 3. DIRECT PIP INSTALL (This bypasses any Conda / requirements file errors)
echo "Installing Python dependencies..."
pip install torch torchvision transformers datasets rasterio evaluate scikit-learn numpy

# 4. RUN SCRIPT
echo "Running Python script..."
python DINOV2Segmentation.py \
    --hf_token "${HF_TOKEN}" \
    --tif_url "${TIF_URL}" \
    --model_path "./model/model.pt" \
    --output_dir "${OUTPUT_DIR}" \
    --threshold "${THRESHOLD}"