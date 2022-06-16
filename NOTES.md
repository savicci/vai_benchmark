### Installing tensorflow and tfds on python 3.6 which vaitrace uses
* /usr/bin/python3 -m pip install tensorflow_datasets==2.1.0
* /usr/bin/python3 -m pip install tensorflow==1.15.5

### Latency and throughput DPUs

Throughput works better for multiple threads, but slower for 1 example, like [images[0]]. ENGINE4 dpu (default) 
can process 4 inferences at once (batch size 4)

Latency has only 1 engine, slower for multiple threads, faster for 1 example. 

Fmnist 1 example
0.0018ms for throughput, 0.0012ms for latency