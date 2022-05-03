#!/usr/bin/bash

# activate vitis-ai-first like ~/Vitis-AI/docker_run.sh xilinx/vitis-ai-gpu:latest
# this script must be run in vitis docker environment
conda activate vitis-ai-optimizer_tensorflow2
pip3 install tensorflow_datasets

export XILINXD_LICENSE_FILE=/workspace/home/gkoziol/vai_optimizer.lic