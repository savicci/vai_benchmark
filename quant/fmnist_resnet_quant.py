import sys

sys.path.append('../bench_perf')

import fmnist_utils
import tensorflow as tf
import argparse
import os
from contextlib import redirect_stdout
from tensorflow_model_optimization.quantization.keras import vitis_quantize

input_shape = (28, 28, 1)
output_shape = 10

CALIB_BATCH_SIZE = 10

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


def app(batch_size, epochs, path, model_path):
    ds_train, ds_test = fmnist_utils.load_dataset(batch_size)

    # trained model
    model = tf.keras.models.load_model(model_path)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    quantizer = vitis_quantize.VitisQuantizer(model)

    # quantize with fine-tuning
    quantized_model_ft = quantizer.quantize_model(calib_dataset=ds_train, calib_steps=10, calib_batch_size=10,
                                                  include_fast_ft=True, fast_ft_epochs=5)

    # quantize without fine-tuning
    quantized_model_no_ft = quantizer.quantize_model(calib_dataset=ds_train, calib_steps=10, calib_batch_size=10,
                                                     include_fast_ft=False)

    # quantization aware training
    quantizer_qat = vitis_quantize.VitisQuantizer(model, quantize_strategy='8bit_tqt')

    qat_model = quantizer_qat.get_qat_model(init_quant=True, calib_dataset=ds_train)
    qat_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    qat_model.fit(ds_train, epochs=epochs)

    qat_quantized_model = quantizer_qat.get_deploy_model(model)

    # record evaluation results
    quantized_model_no_ft.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    with open('{}/quant_no_ft_evaluate.txt'.format(path), 'w') as f:
        with redirect_stdout(f):
            quantized_model_no_ft.evaluate(ds_test)

    quantized_model_ft.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    with open('{}/quant_ft_evaluate.txt'.format(path), 'w') as f:
        with redirect_stdout(f):
            quantized_model_ft.evaluate(ds_test)

    qat_quantized_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    with open('{}/quant_qat_evaluate.txt'.format(path), 'w') as f:
        with redirect_stdout(f):
            qat_quantized_model.evaluate(ds_test)

    # save
    quantized_model_no_ft.save('{}/quant_no_ft.h5'.format(path))
    quantized_model_ft.save('{}/quant_no_ft.h5'.format(path))
    qat_quantized_model.save('{}/quant_no_ft.h5'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default='128',
                        help='Batch size to use for QAT training. Default is 128')
    parser.add_argument('-e', '--epochs', type=int, default='10',
                        help='Epoch number to train QAT network. Default is 10')
    parser.add_argument('-p', '--path', type=str, default='/workspace/vai_benchmark/data/resnet_fmnist/quant',
                        help='Path to folder where all information will be written. Default '
                             '/workspace/vai_benchmark/data/resnet_fmnist/quant')
    parser.add_argument('-m', '--model', type=str,
                        default='/workspace/vai_benchmark/data/resnet_fmnist/trained/init_trained_model.h5',
                        help='Path to trained model. Default /workspace/vai_benchmark/data/resnet_fmnist/trained'
                             '/init_trained_model.h5')

    args = parser.parse_args()
    print('Command line options:')
    print(' --batch_size   : ', args.batch_size)
    print(' --epochs   : ', args.epochs)
    print(' --path   : ', args.path)
    print(' --model   : ', args.model)

    # create dir
    os.makedirs(args.path, exist_ok=True)

    app(args.batch_size, args.epochs, args.path, args.model)
