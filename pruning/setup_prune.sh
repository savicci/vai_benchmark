#!/usr/bin/bash

# activate vitis-ai-first like ~/Vitis-AI/docker_run.sh xilinx/vitis-ai-gpu:latest
# this script must be run in vitis docker environment

pip3 install tensorflow_datasets

conda activate vitis-ai-optimizer_tensorflow2

export XILINXD_LICENSE_FILE=/workspace/vai_optimizer.lic