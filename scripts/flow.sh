#!/usr/bin/bash

# create random prefix
PREFIX = $(head /dev/urandom | tr -dc A-Za-z0-9 | head -c10)
echo 'Using prefix $PREFIX'

RATIO=0.9
CALIBRATION_STEPS=100
WORKSPACE=/workspace/bechmark_results

# compile for throughput
ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json


# use tf2 env
# train network
conda activate vitis-ai-tensorflow2
python ../train/fmnist_custom_train.py --batch_size 128 --epochs 20 --workspace $WORKSPACE --prefix $PREFIX

# setup for pruning
# prune network
conda activate vitis-ai-optimizer_tensorflow2
export XILINXD_LICENSE_FILE=/workspace/vai_optimizer.lic
python ../pruning/fmnist_custom_prune.py --batch_size 128 --epochs 10 --workspace $WORKSPACE --prefix $PREFIX --ratio $RATIO

# setup for quantizing
# quantize network
conda activate vitis-ai-tensorflow2
python ../quant/fmnist_custom_quantize.py --batch_size 10 --epochs 10 --workspace $WORKSPACE --prefix $PREFIX --calibrations $CALIBRATION_STEPS

# compile network
vai_c_tensorflow2 -m $WORKSPACE/$PREFIX/quantized/fmnist_model.h5 -a $ARCH -o $WORKSPACE/$PREFIX/compiled -n fmnist