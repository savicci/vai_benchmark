#!/usr/bin/bash

# use high throughput architecture
ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json

# loop so we have more data
for i in 1 2 3
do
  # prepare model
  conda activate vitis-ai-tensorflow2
  python ./prepare_model.py --layers 1

  # compile
  vai_c_tensorflow2 -m ./fmnist_model.h5 -a $ARCH -o ./compiled -n fmnist

  # run test with profiler
  python ../test/dpu_single.py -m vaitrace_py --model $(pwd)/compiled/fmnist.xmodel
done

# collect data for further processing
python ./collect_data.py