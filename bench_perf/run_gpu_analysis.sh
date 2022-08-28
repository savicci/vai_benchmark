#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=1

conda activate vitis-ai-tensorflow2

for i in {1..40}
do
  echo Running $i loop

  python ../test/gpu_single_custom.py --layer $i --file gpu_results.csv
done