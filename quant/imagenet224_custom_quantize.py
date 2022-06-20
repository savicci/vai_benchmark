import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
from contextlib import redirect_stdout
from tensorflow_model_optimization.quantization.keras import vitis_quantize
import os


def add_normalized_values(img, label):
    """Normalizes images"""
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return tf.image.resize(norm_img, [224, 224]), label


def load_dataset(batch_size):
    (ds_train, ds_validation) = tfds.load('imagenet2012', split=['train', 'validation'], as_supervised=True, shuffle_files=True)
    # map data
    ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_validation = ds_validation.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # use batching
    ds_validation = ds_validation.batch(batch_size)
    ds_train = ds_train.batch(batch_size)

    return ds_train, ds_validation


def load_model(workspace, prefix, network) -> tf.keras.models.Model:
    return tf.keras.models.load_model(workspace + '/' + prefix + '/pruned/{}_model'.format(network))


def app(workspace, calibrations, prefix, batch_size, fast_ft_epochs, network):
    ds_train, ds_test = load_dataset(batch_size)

    # model to use
    model = load_model(workspace, prefix, network)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # create quantizer
    quantizer = vitis_quantize.VitisQuantizer(model)

    # quantize with fine tuning
    quantized_model = quantizer.quantize_model(calib_dataset=ds_train, calib_steps=calibrations,
                                               calib_batch_size=batch_size, include_fast_ft=True,
                                               fast_ft_epochs=fast_ft_epochs)
    quantized_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # evaluate
    with open(workspace + '/' + prefix + '/quantized/{}_pruned_evaluate.txt'.format(network), 'w+') as f:
        with redirect_stdout(f):
            loss, accuracy = quantized_model.evaluate(ds_test, verbose=2)
            print('Loss {}, accuracy {}'.format(loss, accuracy))

    # save init summary
    with open(workspace + '/' + prefix + '/quantized/{}_init_summary.txt'.format(network), 'w+') as f:
        with redirect_stdout(f):
            quantized_model.summary()

    # save model
    quantized_model.save(workspace + '/' + prefix + '/quantized/{}_model.h5'.format(network))

    print('Finished quantizing and saving information')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default='10',
                        help='Batch size to use for calibration. Default is 10')
    parser.add_argument('-c', '--calibrations', type=int, default='100',
                        help='Calibration steps to use while quantizing. Default is 100')
    parser.add_argument('-w', '--workspace', type=str, default='/workspace/results',
                        help='Path to folder to write pruned model summary, evaluate and h5')
    parser.add_argument('-p', '--prefix', type=str, default='default',
                        help='Prefix to folder where all information will be written')
    parser.add_argument('-f', '--fast_ft_epochs', type=int, default='10',
                        help='Amount of epochs for fast finetuning quantized model')
    parser.add_argument('-n', '--network', type=str, default='custom',
                        help='Name of network. App uses this prop to save and load files')

    args = parser.parse_args()
    print('Command line options:')
    print(' --workspace             : ', args.workspace)
    print(' --calibrations          : ', args.calibrations)
    print(' --prefix                : ', args.prefix)
    print(' --batch_size            : ', args.batch_size)
    print(' --fast_ft_epochs        : ', args.fast_ft_epochs)
    print(' --network               : ', args.network)

    # create dir
    os.makedirs(args.workspace + '/' + args.prefix + '/quantized', exist_ok=True)

    app(args.workspace, args.calibrations, args.prefix, args.batch_size, args.fast_ft_epochs, args.network)
