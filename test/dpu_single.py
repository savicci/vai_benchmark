from typing import List, Tuple

import numpy as np
import xir
# from vaitrace_py import vai_tracepoint
import vart
import argparse
import time
import fmnist_utils
import os
import csv

divider = '------------------------------------'


# vai_tracepoint decorator enables tracing on function level
# @vai_tracepoint
def load_tensorflow_dataset() -> Tuple[List, List]:
    return fmnist_utils.load_tensorflow_dataset()


# @vai_tracepoint
def preprocess_dataset(images, scale) -> List:
    return fmnist_utils.preprocess_dataset(images, scale)


# @vai_tracepoint
def postprocess_results(out_vectors, labels):
    fmnist_utils.postprocess_results(out_vectors, labels)


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


def run_dpus(dpu_runners, img):
    index = 0

    img_size = len(img)
    dpu_cores = len(dpu_runners)

    dpu_input_ndims = []
    dpu_output_ndims = []

    for runner in dpu_runners:
        dpu_input_ndims.append(tuple(runner.get_input_tensors()[0].dims))
        dpu_output_ndims.append(tuple(runner.get_output_tensors()[0].dims))

    print("Input tensors dimensions", dpu_input_ndims)
    print("Output tensors dimensions", dpu_output_ndims)

    output_data = []
    jobs = []
    iteration = 0

    while index < img_size:
        max_batch_size = dpu_input_ndims[iteration % dpu_cores][0]

        # determine batch size
        if index + max_batch_size > img_size:
            batch_size = img_size - index
        else:
            batch_size = max_batch_size

        # create space for dpu output
        output_data.append([np.empty(dpu_output_ndims[iteration % dpu_cores], dtype=np.int8)])

        # fill batch input data
        input_data = [np.empty(dpu_input_ndims[iteration % dpu_cores], dtype=np.int8)]

        # initialize input image to input buffer
        for i in range(batch_size):
            img_data = input_data[0]
            img_data[i, ...] = img[index + i].reshape(dpu_input_ndims[iteration % dpu_cores][1:])

        # run job asynchronously
        job_id = dpu_runners[iteration % dpu_cores].execute_async(input_data, output_data[len(jobs)])
        jobs.append((job_id, batch_size, index))

        # post - update indexes
        index += batch_size
        iteration += 1

    # process output data after jobs finish running
    jobs_running = len(jobs)
    for i in range(jobs_running):
        dpu_runners[i % dpu_cores].wait(jobs[i][0])

        write_index = jobs[i][2]

        # store output vectors in global variable
        for j in range(jobs[i][1]):
            # get top 1 value
            output_vectors[write_index] = np.argmax(output_data[i][0][j])
            write_index += 1


def app(model, dpu_cores, file, layer):
    # load dataset
    images, labels = load_tensorflow_dataset()
    print('Loaded dataset')

    # create global variable output vectors for runDPU function to fill with data after processing
    global output_vectors
    output_vectors = [None] * len(images)

    # deserialize xilinx xmodel to graph
    graph = xir.Graph.deserialize(model)

    # graph partitions
    sub_graphs = get_child_subgraph_dpu(graph)

    # Creates an instance of CPU/SIM/DPU runner by subgraph. Run is DPU runner
    dpu_runners = []
    for i in range(dpu_cores):
        dpu_runners.append(vart.Runner.create_runner(sub_graphs[0], "run"))

    # input scaling
    input_fixpos = dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2 ** input_fixpos

    print('Preprocessing {} images'.format(len(images)))
    processed_images = preprocess_dataset(images, input_scale)

    start_time = time.time()
    run_dpus(dpu_runners, processed_images)
    end_time = time.time()

    execution_time = end_time - start_time

    throughput = float(len(processed_images) / execution_time)
    print(divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" % (
        throughput, len(processed_images), execution_time))

    # write header row
    header_row = ['Params', 'Throughput [fps]', 'Execution time [s]']
    if not os.path.exists(file):
        with open(file, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header_row)

    with open('/workspace/vai_benchmark/bench_perf/params_{}.txt'.format(layer), 'r') as f:
        params = f.readline()

    data_row = [params, throughput, execution_time]
    with open(file, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(data_row)

    # print(divider)
    # print('Postprocessing {} images'.format(len(processed_images)))
    # postprocess_results(output_vectors, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, default='file.xmodel',
                        help='Path to xmodel file. Default is file.xmodel in current directory')
    parser.add_argument('-d', '--dpu_cores', type=int, default=3,
                        help='DPU cores to use. Default for Alveo U280 is 3')
    parser.add_argument('-f', '--file', type=str, default='dpu_results.csv',
                        help='File to append result data to. Default is dpu_results.csv')
    parser.add_argument('-l', '--layer', type=str, default='dpu_results.csv',
                        help='File to append result data to. Default is dpu_results.csv')

    args = parser.parse_args()
    print('Command line options:')
    print(' --model     : ', args.model)
    print(' --dpu_cores     : ', args.dpu_cores)
    print(' --file     : ', args.file)
    print(' --layer     : ', args.layer)

    app(args.model, args.dpu_cores, args.file, args.layer)
