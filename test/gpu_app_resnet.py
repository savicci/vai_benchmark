import argparse
import time

import numpy as np
import tensorflow as tf

import resnet_utils

divider = '------------------------------------'


def app(model, batch_size, threads, size):
    tf.config.experimental.list_physical_devices()

    # load dataset
    images, labels = resnet_utils.load_tensorflow_dataset(size)

    # load model
    model = tf.keras.models.load_model(model)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # to keep same interfaces, will not do anything
    input_scale = 1

    print('Preprocessing {} images'.format(len(images)))
    processed_images = resnet_utils.preprocess_dataset(images, input_scale)
    # convert to numpy arrays
    processed_images = np.array(processed_images)

    start_time = time.time()

    # process images
    output_vectors = model.predict(processed_images, batch_size=batch_size, workers=threads, use_multiprocessing=True)

    # get top 1 value
    output_vectors = [np.argmax(prediction) for prediction in output_vectors]

    end_time = time.time()

    execution_time = end_time - start_time

    throughput = float(len(processed_images) / execution_time)
    print(divider)
    print("Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds" % (
        throughput, len(processed_images), execution_time))

    print(divider)
    print('Postprocessing {} images'.format(len(processed_images)))
    resnet_utils.postprocess_results(output_vectors, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, default='model.h5',
                        help='Path to h5 model file. Default is model.h5 in current directory')
    parser.add_argument('-b', '--batch_size', type=int, default='32',
                        help='Batch size used for prediction. Default is 32')
    parser.add_argument('-t', '--threads', type=int, default='1',
                        help='Workers to use for multi process threading prediction. Default is 1')
    parser.add_argument('-s', '--size', type=int, default=1000,
                        help='Size of validation dataset to use. Default is 1k')

    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    args = parser.parse_args()
    print(' --model     : ', args.model)
    print(' --batch_size     : ', args.batch_size)
    print(' --threads     : ', args.threads)
    print(' --size     : ', args.size)

    app(args.model, args.batch_size, args.threads, args.size)
