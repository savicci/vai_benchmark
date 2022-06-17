#!/usr/bin/bash

# create random prefix
PREFIX=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c10)
echo Using prefix $PREFIX

RATIO=0.3
CALIBRATION_STEPS=100
WORKSPACE=/workspace/bechmark_results

# export visible device for tf
export CUDA_VISIBLE_DEVICES=1

# compile for throughput
ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json

# use tf2 env
# train network
conda activate vitis-ai-tensorflow2
python ../train/resnet_custom_train.py --batch_size 40 --epochs 10 --workspace $WORKSPACE --prefix $PREFIX

# setup for pruning
# prune network
conda activate vitis-ai-optimizer_tensorflow2
export XILINXD_LICENSE_FILE=/workspace/vai_optimizer.lic
python ../pruning/resnet_custom_prune.py --batch_size 40 --epochs 2 --workspace $WORKSPACE --prefix $PREFIX --ratio $RATIO

# setup for quantizing
# quantize network
conda activate vitis-ai-tensorflow2
python ../quant/resnet_custom_quantize.py --batch_size 10 --fast_ft_epochs 10 --workspace $WORKSPACE --prefix $PREFIX --calibrations $CALIBRATION_STEPS

# compile network
vai_c_tensorflow2 -m $WORKSPACE/$PREFIX/quantized/resnet_model.h5 -a $ARCH -o $WORKSPACE/$PREFIX/compiled -n resnet

echo Finished compiling file in $WORKSPACE/$PREFIX