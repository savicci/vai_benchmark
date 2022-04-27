# Vitis AI Benchmarking repo for master's degree calculations

## Uses ML Perf benchmarks. Test must be done on local gpu (rtx 3060) and on xilinx DPU on Alveo u280

### MlPerf - 99% accuracy on FP32

### Models to test

* Image classification - ResNet with ImageNet dataset
* Object detection (large) - SSD-Large with COCO dataset
* Medical imaging - 3D-UNet with BraTS 2019 dataset
* Speech-to-text - RNN-T with LibriSpeech dataset
* NLP - BERT with SQuAD v1.1 dataset
* Recommendation - DLRM with Criteo 1Tb dataset 

### installing env for tensorflow2
* conda create -n tf_gpu_env python=3.8 
* conda activate tf_gpu_env
* conda install tensorflow-gpu -c anaconda
* conda install cudnn -c conda-forge 
* conda install cudatoolkit -c anaconda
* pip install --upgrade tensorflow
* pip install jupyter
* pip install pillow