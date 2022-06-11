#!/usr/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Correct usage: source compileLatency.sh <PATH_TO_MODEL_H5> <COMPILED_BUILD_NAME> <NAME OF NET>"
    exit
fi


PATH_TO_MODEL=$1
TARGET=$2
NETNAME=$3

ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8L/U280/arch.json

vai_c_tensorflow2 -m $PATH_TO_MODEL -a $ARCH -o build/compiled_$TARGET -n $NETNAME