#!/usr/bin/bash

# install tfds
pip3 install tensorflow_datasets

# save datasets on persistent storage instead of in container
export TFDS_DATA_DIR=/workspace/storage/gkoziol/tensorflow_datasets