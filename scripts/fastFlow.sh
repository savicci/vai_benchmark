#!/usr/bin/bash

# create random prefix
PREFIX=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c10)
echo Using prefix $PREFIX

RATIO=0.3
CALIBRATION_STEPS=100
WORKSPACE=/workspace/bechmark_results

if [ $# -ne 3 ]; then
     echo "Correct usage: ./fastFlow.sh <model> <network> <cuda_visible_devices>"
     echo "Example: ./fastFlow.sh resnet_50.sh resnet 0,1"
     return 1 2>/dev/null
     exit 1
fi

# parameters for flow (train, prune, quantize, compile)
MODEL=$1
NETWORK=$2

# export visible device for tf
export CUDA_VISIBLE_DEVICES=$3

echo "Running with params:"
echo model: $MODEL
echo network: $NETWORK
echo cuda_visible_devices: $CUDA_VISIBLE_DEVICES

# compile for throughput
ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json

# use tf2 env
# train network
source /opt/vitis_ai/conda/etc/profile.d/conda.sh

conda activate vitis-ai-tensorflow2
# will start training from pretrained model. If you want to supply your own override --checkpoint /workspace/bechmark_results/lm6NcbXDQf/trained/some_model
python ../train/imagenet224_custom_train.py --batch_size 64 --epochs 1 --workspace $WORKSPACE --prefix $PREFIX --model $MODEL --network $NETWORK

# setup for pruning
# prune network
conda activate vitis-ai-optimizer_tensorflow2
export XILINXD_LICENSE_FILE=/workspace/vai_optimizer.lic
python ../pruning/imagenet224_custom_prune.py --batch_size 64 --epochs 1 --workspace $WORKSPACE --prefix $PREFIX --ratio $RATIO --network $NETWORK

# setup for quantizing
# quantize network
conda activate vitis-ai-tensorflow2
python ../quant/imagenet224_custom_quantize.py --batch_size 5 --fast_ft_epochs 5 --workspace $WORKSPACE --prefix $PREFIX --calibrations $CALIBRATION_STEPS --network $NETWORK

# compile network
vai_c_tensorflow2 -m $WORKSPACE/$PREFIX/quantized/$NETWORK_model.h5 -a $ARCH -o $WORKSPACE/$PREFIX/compiled -n $NETWORK

echo Finished compiling file in $WORKSPACE/$PREFIX