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