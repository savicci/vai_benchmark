#!/usr/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Correct usage: source compileThroughput.sh <PATH_TO_MODEL_H5> <COMPILED_BUILD_NAME> <NAME OF NET>"
    exit
fi

PATH_TO_MODEL=$1
OUTPUTPATH=$2
NETNAME=$3

ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json

vai_c_tensorflow2 -m $PATH_TO_MODEL -a $ARCH -o $OUTPUTPATH -n $NETNAME