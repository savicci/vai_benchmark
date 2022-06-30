#!/usr/bin/bash

# use high throughput architecture
ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json

conda activate vitis-ai-tensorflow2

# loop so we have more data
for i in 1 2 3
do
  # run test with profiler
  python ../test/dpu_single.py -m vaitrace_py --model $(pwd)/compiled_$i/fmnist.xmodel

  # collect data for further processing
  python ./collect_data.py --layer $i
done