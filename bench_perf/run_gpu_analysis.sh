#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=1

conda activate vitis-ai-tensorflow2

rm -f gpu_results.csv

for i in {1..40}
do
  echo Running $i loop

  python ../test/gpu_single_custom.py --layers $i --file gpu_results.csv --batch_size 128 --threads 3
done