import tensorflow as tf
import fmnist_utils
import argparse
from tensorflow_model_optimization.quantization.keras import vitis_quantize
import numpy as np

# variables
epochs = 1
calibrations = 10


def create_model(layers_num):
    layers = []

    layers.extend([
        tf.keras.Input(shape=fmnist_utils.shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
    ])

    # for i in range(layers_num):
    #     layers.append(tf.keras.layers.Dense(1000, activation="relu"))

    layers.extend([
        tf.keras.layers.Flatten(), tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(fmnist_utils.output_classes, activation="softmax"),
    ])

    return tf.keras.Sequential(layers)


def app(batch_size, layers):
    # load dataset
    ds_train, ds_test = fmnist_utils.load_dataset(batch_size)

    # create model
    model = create_model(layers)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # train for a moment
    model.fit(ds_train, epochs=epochs)

    model.save('./fmnist_temp')
    model = tf.keras.models.load_model('./fmnist_temp')
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # quantize without pruning
    quantizer = vitis_quantize.VitisQuantizer(model)
    quantized_model = quantizer.quantize_model(calib_dataset=ds_train, calib_steps=calibrations)

    # save model
    quantized_model.save('./fmnist_model.h5')

    # save param number
    params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    with open('./params_{}.txt'.format(layers), 'w') as f:
        f.write(str(params))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default='128',
                        help='Batch size to use for training. Default is 128')
    parser.add_argument('-l', '--layers', type=int, default=1,
                        help='Amount of dense layers with 1k output size. Default is 1')

    args = parser.parse_args()
    print('Command line options:')
    print(' --batch_size        : ', args.batch_size)
    print(' --layers            : ', args.layers)

    app(args.batch_size, args.layers)
