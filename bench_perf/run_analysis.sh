#!/usr/bin/bash

# run on vitis 1.4.1 container because on other version dpu does not answer
source ../compile/setupAlveoThroughput.sh

conda activate vitis-ai-tensorflow2

# loop so we have more data
for ((i=200;i<=510;i=i+10))
do
  echo Running $i loop

  # run test with profiler
  python -m vaitrace_py -t 30 ../test/dpu_single.py --model $(pwd)/compiled_$i/fmnist.xmodel

  # collect data for further processing
  python ./collect_data.py --layer $i

  sleep 3
done