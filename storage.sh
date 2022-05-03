#!/usr/bin/bash

# install tfds
conda activate vitis-ai-optimizer_tensorflow2
pip3 install tensorflow_datasets

conda activate vitis-ai-tensorflow2
pip3 install tensorflow_datasets

# save datasets on persistent storage instead of in container
export TFDS_DATA_DIR=/workspace/storage/tensorflow_datasets