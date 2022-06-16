import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
from contextlib import redirect_stdout
from tensorflow_model_optimization.quantization.keras import vitis_quantize
import os

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

tf.config.experimental.set_visible_devices(physical_devices[1], 'GPU')


def add_normalized_values(img, label):
    """Normalizes images"""
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return norm_img, label


def load_dataset(batch_size):
    (ds_train, ds_test) = tfds.load('fashion_mnist', split=['train', 'test'], as_supervised=True, shuffle_files=True)

    # use batching
    ds_test = ds_test.batch(batch_size)
    ds_train = ds_train.batch(batch_size)

    # map data
    ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test


def load_model(workspace, prefix) -> tf.keras.models.Model:
    return tf.keras.models.load(workspace + '/' + prefix + '/pruned/fmnist_model.h5')


def app(epochs, workspace, calibrations, prefix, batch_size):
    # model to use
    model = load_model(workspace, prefix)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # create quantizer
    quantizer = vitis_quantize.VitisQuantizer(model)

    # quantize with fine tuning
    quantized_model = quantizer.quantize_model(calib_dataset=ds_train, calib_steps=calibrations, calib_batch_size=batch_size)

    # fine-tuning process
    quantized_model.fit(ds_train, epochs=epochs)

    # evaluate
    with open(workspace + '/' + prefix + '/quantized/fmnist_pruned_evaluate.txt', 'w+') as f:
        with redirect_stdout(f):
            loss, accuracy = quantized_model.evaluate(ds_test, verbose=2)
            print('Loss {}, accuracy {}'.format(loss, accuracy))

    # save init summary
    with open(workspace + '/' + prefix + '/quantized/fmnist_init_summary.txt', 'w+') as f:
        with redirect_stdout(f):
            quantized_model.summary()

    # save model
    quantized_model.save(workspace + '/' + prefix + '/quantized/fmnist_model.h5')

    print('Finished quantizing and saving information')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default='10',
                        help='Batch size to use for calibration. Default is 10')
    parser.add_argument('-e', '--epochs', type=int, default='10',
                        help='Epoch number to finetune network. Default is 10')
    parser.add_argument('-c', '--calibrations', type=int, default='100',
                        help='Calibration steps to use while quantizing. Default is 100')
    parser.add_argument('-w', '--workspace', type=str, default='/workspace/results',
                        help='Path to folder to write pruned model summary, evaluate and h5')
    parser.add_argument('-p', '--prefix', type=str, default='default',
                        help='Prefix to folder where all information will be written')

    args = parser.parse_args()
    print('Command line options:')
    print(' --batch_size   : ', args.batch_size)
    print(' --epochs   : ', args.epochs)
    print(' --workspace   : ', args.workspace)
    print(' --prefix   : ', args.prefix)

    # load dataset. Needs to be done earlier for evaluate function
    ds_train, ds_test = load_dataset(args.batch_size)

    # create dir
    os.makedirs(args.workspace + '/' + args.prefix + '/quantized', exist_ok=True)

    app(args.epochs, args.workspace, args.calibrations, args.prefix, args.batch_size)
