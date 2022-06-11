from typing import List, Tuple

import numpy as np
import xir
from vaitrace_py import vai_tracepoint
import vart
import argparse
import threading
import time
import tensorflow_datasets as tfds

divider = '------------------------------------'


# vai_tracepoint decorator enables tracing on function level
@vai_tracepoint
def load_tensorflow_dataset() -> Tuple[List, List]:
    ds_test = tfds.load('fashion_mnist', split='test', shuffle_files=True)

    images = []
    labels = []
    for record in tfds.as_numpy(ds_test):
        images.append(record['image'])
        labels.append(record['label'])

    return images, labels


@vai_tracepoint
def preprocess_dataset(images, scale) -> List:
    return [record * (1 / 255.0) * scale for record in images]


@vai_tracepoint
def postprocess_results(out_vectors, labels):
    correct = 0
    miss = 0

    for i in range(len(out_vectors)):
        prediction = out_vectors[i]

        if prediction == labels[i]:
            correct += 1
        else:
            miss += 1

    accuracy = correct / len(out_vectors)
    print('Correct:%d, Wrong:%d, Accuracy:%.4f' % (correct, miss, accuracy))
    print(divider)


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."

    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."

    if root_subgraph.is_leaf:
        return []

    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0

    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(id, start, dpu, img):
    # get input/output tensors from DPU
    input_tensors = dpu.get_input_tensors()
    output_tensors = dpu.get_output_tensors()

    # tensor dimensions
    input_ndim = tuple(input_tensors[0].dims)
    output_ndim = tuple(output_tensors[0].dims)

    batch_size = input_ndim[0]
    n_of_images = len(img)
    count = 0
    ids = []
    ids_max = 50
    outputData = []
    for i in range(ids_max):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
    while count < n_of_images:
        if (count + batch_size <= n_of_images):
            runSize = batch_size
        else:
            runSize = n_of_images - count

        '''prepare batch input/output '''
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])
        '''run with batch '''
        job_id = dpu.execute_async(inputData, outputData[len(ids)])
        ids.append((job_id, runSize, start + count))
        count = count + runSize
        if count < n_of_images:
            if len(ids) < ids_max - 1:
                continue
        for index in range(len(ids)):
            dpu.wait(ids[index][0])
            write_index = ids[index][2]
            '''store output vectors '''
            for j in range(ids[index][1]):
                # we can avoid output scaling if use argmax instead of softmax
                # out_q[write_index] = np.argmax(outputData[0][j] * output_scale)
                output_vectors[write_index] = np.argmax(outputData[index][0][j])
                write_index += 1
        ids = []
    pass


def app(model, threads):
    # load dataset
    images, labels = load_tensorflow_dataset()

    # create global variable output vectors for runDPU function to fill with data after processing
    global output_vectors
    output_vectors = [None] * len(images)

    # deserialize xilinx xmodel to graph
    graph = xir.Graph.deserialize(model)

    # graph partitions
    sub_graphs = get_child_subgraph_dpu(graph)

    dpu_runners = []
    for i in range(threads):
        # Creates an instance of CPU/SIM/DPU runner by subgraph. Run is DPU runner
        dpu_runners.append(vart.Runner.create_runner(sub_graphs[0], "run"))

    # input scaling
    input_fixpos = dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2 ** input_fixpos

    print('Preprocessing {} images'.format(len(images)))
    processed_images = preprocess_dataset(images, input_scale)

    print('Running {} threads'.format(threads))
    all_threads = []
    start = 0

    # split images into batches for DPU cause it can process more than 1 at a time
    for i in range(threads):
        # split processing of images into threads_num batches
        if i == threads - 1:
            end = len(processed_images)
        else:
            end = start + (len(processed_images) // threads)

        input_vectors = processed_images[start:end]
        thread = threading.Thread(target=runDPU, args=(i, start, dpu_runners[i], input_vectors))
        all_threads.append(thread)
        start = end

    start_time = time.time()
    # start processing
    for thread in all_threads:
        thread.start()

    # wait for all
    for thread in all_threads:
        thread.join()
    end_time = time.time()
    execution_time = end_time - start_time

    throughput = float(len(processed_images) / execution_time)
    print(divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" % (
        throughput, len(processed_images), execution_time))

    print('Postprocessing {} images'.format(len(processed_images)))
    postprocess_results(output_vectors, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, default='file.xmodel',
                        help='Path to xmodel file. Default is file.xmodel in current directory')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of threads running at once. Includes pre/post processing and DPU runners. '
                             'Default is 1')

    args = parser.parse_args()
    print('Command line options:')
    print(' --threads   : ', args.threads)
    print(' --model     : ', args.model)

    app(args.model, args.threads)
