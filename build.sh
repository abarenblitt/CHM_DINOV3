#!/bin/bash
set -x
basedir=$( cd "CHM_DINOv3/notebooks" ; pwd -P )

####install requirements packages
conda env update -f ${basedir}/requirements.yml
#pushd ${HOME}
#source activate dino_envi