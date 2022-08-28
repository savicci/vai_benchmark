import tensorflow as tf
import fmnist_utils
import argparse
from tensorflow_model_optimization.quantization.keras import vitis_quantize
import numpy as np
import resnet_seq

# variables
epochs = 1
calibrations = 10


def app(batch_size, layers):
    # load dataset
    ds_train, ds_test = fmnist_utils.load_dataset(batch_size)

    # create model
    # model = create_model(layers)
    model = resnet_seq.customized_resnet((28, 28, 1), 10, layers)
    model.summary()

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # train for a moment
    # model.fit(ds_train, epochs=epochs)

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
    parser.add_argument('-l', '--layers', type=int, default=40,
                        help='Amount of residual blocks. Default is 1')

    args = parser.parse_args()
    print('Command line options:')
    print(' --batch_size        : ', args.batch_size)
    print(' --layers            : ', args.layers)

    app(args.batch_size, args.layers)
