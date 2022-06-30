#!/usr/bin/bash

# run on vitis 2.0 container cause of problems with quantizer

# use high throughput architecture
ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json

conda activate vitis-ai-tensorflow2

# loop so we have more data
for i in 1 2 3
do
  # prepare model
  python prepare_model.py --layers $i

  # compile
  vai_c_tensorflow2 -m ./fmnist_model.h5 -a $ARCH -o ./compiled_$i -n fmnist
done