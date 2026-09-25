#!/bin/bash
set -e

# 1. Catch positional arguments automatically sent by MAAP
HF_TOKEN=$1

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
python training.py \
    --hf_token "${HF_TOKEN}" \
    --output_dir "${OUTPUT_DIR}" \
    --threshold "${THRESHOLD}"