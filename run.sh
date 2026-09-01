#!/bin/bash

# Get the directory where this run.sh script lives
basedir=$( cd "$(dirname "$0")" ; pwd -P )

# MAAP requires outputs to go to a specific output folder in the execution directory
mkdir -p output
outdir=${PWD}/output

# Change to the directory where your code lives
cd ${basedir}

# Run the python script
# MAAP automatically passes config inputs (like HF_TOKEN) as Environment Variables
python3 DINOV2Segmentation.py