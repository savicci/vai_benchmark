#!/usr/bin/bash

# run on vitis 1.4.1 container because on other version dpu does not answer
source ../compile/setupAlveoLatency.sh

# use high throughput architecture
ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8L/U280/arch.json

conda activate vitis-ai-tensorflow2

# loop so we have more data
for i in {1..40}
do
  echo Running $i loop

  # run test with profiler
  python -m vaitrace_py ../test/dpu_single.py --model $(pwd)/compiled_$i/fmnist.xmodel

  # collect data for further processing
  python ./collect_data.py --layer $i

  sleep 3
done