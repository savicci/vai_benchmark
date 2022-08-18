import tensorflow_datasets as tfds
import tensorflow as tf
import argparse
import os
from contextlib import redirect_stdout
from tensorflow_model_optimization.quantization.keras import vitis_quantize

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


def add_normalized_values(img, label):
    """Normalizes images"""
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return tf.image.resize(norm_img, [224, 224]), label


def load_dataset(batch_size):
    (ds_train, ds_validation) = tfds.load('imagenet2012', split=['train', 'validation'], as_supervised=True,
                                          shuffle_files=True)
    # map data
    ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_validation = ds_validation.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # use batching
    ds_validation = ds_validation.batch(batch_size)
    ds_train = ds_train.batch(batch_size)
    return ds_train, ds_validation


def app(batch_size, epochs, path, model_path):
    # for ptq
    ds_train, ds_test = load_dataset(16)

    # trained model
    model = tf.keras.models.load_model(model_path)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    no_ft_quantizer = vitis_quantize.VitisQuantizer(model)
    ft_quantizer = vitis_quantize.VitisQuantizer(model)
    quantizer_qat = vitis_quantize.VitisQuantizer(model, quantize_strategy='8bit_tqt')

    # quantize without fine-tuning
    print("Start quantizing not ft")
    quantized_model_no_ft = no_ft_quantizer.quantize_model(calib_dataset=ds_train, calib_steps=2, calib_batch_size=2,
                                                           include_fast_ft=False)
    # quantize with fine-tuning
    print("Start quantizing ft")
    quantized_model_ft = ft_quantizer.quantize_model(calib_dataset=ds_train, calib_steps=2, calib_batch_size=2,
                                                     include_fast_ft=True, fast_ft_epochs=10)

    # quantization aware training
    print("Start quantizing quat")
    ds_train, ds_test = load_dataset(batch_size)
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
    quantized_model_ft.save('{}/quant_ft.h5'.format(path))
    qat_quantized_model.save('{}/quant_qat.h5'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default='64',
                        help='Batch size to use for QAT training. Default is 64')
    parser.add_argument('-e', '--epochs', type=int, default='5',
                        help='Epoch number to train QAT network. Default is 5')
    parser.add_argument('-p', '--path', type=str, default='/workspace/vai_benchmark/data/resnet/quant',
                        help='Path to folder where all information will be written. Default '
                             '/workspace/vai_benchmark/data/resnet/quant')
    parser.add_argument('-m', '--model', type=str,
                        default='/workspace/vai_benchmark/data/resnet/trained/init_trained_model.h5',
                        help='Path to trained model. Default /workspace/vai_benchmark/data/resnet/trained'
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
