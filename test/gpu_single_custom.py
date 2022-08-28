import sys

sys.path.append('../bench_perf')

import argparse
import time

import numpy as np
import tensorflow as tf

import fmnist_utils
import resnet_seq
import fmnist_utils
import csv
import os


divider = '------------------------------------'


def app(batch_size, layers, file, threads):
    tf.config.experimental.list_physical_devices()

    # load dataset
    images, labels = fmnist_utils.load_tensorflow_dataset()

    # load model
    model = resnet_seq.customized_resnet((28, 28, 1), 10, layers)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # to keep same interfaces, will not do anything
    input_scale = 1

    print('Preprocessing {} images'.format(len(images)))
    processed_images = fmnist_utils.preprocess_dataset(images, input_scale)
    # convert to numpy arrays
    processed_images = np.array(processed_images)

    start_time = time.time()

    # process images
    model.predict(processed_images, batch_size=batch_size, workers=threads, use_multiprocessing=True)

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

    params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    data_row = [params, throughput, execution_time]
    with open(file, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(data_row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default='32',
                        help='Batch size used for prediction. Default is 32')
    parser.add_argument('-l', '--layers', type=int, default='1',
                        help='Layers to add parameters. Default is 1')
    parser.add_argument('-t', '--threads', type=int, default='1',
                        help='Threads to multiprocessing. Default is 1')
    parser.add_argument('-f', '--file', type=str, default='gpu_results.csv',
                        help='File to append result data to. Default is gpu_results.csv')

    args = parser.parse_args()
    print(' --batch_size     : ', args.batch_size)
    print(' --layers     : ', args.layers)
    print(' --file     : ', args.file)
    print(' --threads     : ', args.threads)

    app(args.batch_size, args.layers, args.file, args.threads)
