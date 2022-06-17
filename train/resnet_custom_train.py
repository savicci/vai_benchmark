import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import argparse
from contextlib import redirect_stdout
import os


def add_normalized_values(img, label):
    """Normalizes images"""
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return tf.image.resize(norm_img, [224, 224]), label + 1


def load_dataset(batch_size):
    (ds_train, ds_validation) = tfds.load('imagenet2012', split=['train', 'validation'], as_supervised=True, shuffle_files=True)
    # map data
    ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_validation = ds_validation.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # use batching
    ds_validation = ds_validation.batch(batch_size)
    ds_train = ds_train.batch(batch_size)

    return ds_train, ds_validation


def create_model(input_shape, num_classes) -> tf.keras.models.Model:
    model_handle = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5"

    model = tf.keras.Sequential([
        # Explicitly define the input shape so the model can be properly
        # loaded by the TFLiteConverter
        tf.keras.layers.InputLayer(input_shape=input_shape),
        hub.KerasLayer(model_handle, trainable=True),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,) + input_shape)
    return model


def app(batch_size, epochs, workspace, prefix):
    input_shape = (224, 224, 3)
    num_classes = 1000

    # load dataset
    ds_train, ds_test = load_dataset(batch_size)

    # model to use
    model = create_model(input_shape, num_classes)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # train model
    model.fit(ds_train, batch_size=batch_size, epochs=epochs)

    # evaluate
    with open(workspace + '/' + prefix + '/trained/resnet_evaluate.txt', 'w+') as f:
        with redirect_stdout(f):
            loss, accuracy = model.evaluate(ds_test, verbose=2)
            print('Loss {}, accuracy {}'.format(loss, accuracy))

    # save init summary
    with open(workspace + '/' + prefix + '/trained/resnet_init_summary.txt', 'w+') as f:
        with redirect_stdout(f):
            model.summary()

    # save model
    model.save(workspace + '/' + prefix + '/trained/resnet_model')

    print('Finished training and saving information')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default='32',
                        help='Batch size to use for training. Default is 32')
    parser.add_argument('-e', '--epochs', type=int, default='20',
                        help='Epoch number to train network. Default is 20')
    parser.add_argument('-w', '--workspace', type=str, default='/workspace/results',
                        help='Path to folder to write model summary, evaluate and h5')
    parser.add_argument('-p', '--prefix', type=str, default='default',
                        help='Prefix to folder where all information will be written')

    args = parser.parse_args()
    print('Command line options:')
    print(' --batch_size   : ', args.batch_size)
    print(' --epochs   : ', args.epochs)
    print(' --workspace   : ', args.workspace)
    print(' --prefix   : ', args.prefix)

    # create dir
    os.makedirs(args.workspace + '/' + args.prefix + '/trained', exist_ok=True)

    app(args.batch_size, args.epochs, args.workspace, args.prefix)