#!/usr/bin/bash

cd /

/home/gkoziol/Vitis-AI/docker_run.sh xilinx/vitis-ai-gpu:latest

# save datasets on persistent storage instead of in container
export TFDS_DATA_DIR=/workspace/storage/gkoziol/tensorflow_datasets