import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import argparse
from contextlib import redirect_stdout
import os


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


def create_model(checkpoint, model) -> tf.keras.models.Model:
    if checkpoint is None:
        return tf.keras.models.load_model('../models/{}'.format(model))
    else:
        return tf.keras.models.load_model(checkpoint)


def app(batch_size, epochs, workspace, prefix, checkpoint, model, network):
    # load dataset
    ds_train, ds_test = load_dataset(batch_size)

    # model to use
    model = create_model(checkpoint, model)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # train model
    model.fit(ds_train, validation_data=ds_train, epochs=epochs)

    # evaluate
    with open(workspace + '/' + prefix + '/trained/{}_evaluate.txt'.format(network), 'w+') as f:
        with redirect_stdout(f):
            loss, accuracy = model.evaluate(ds_test, verbose=2)
            print('Loss {}, accuracy {}'.format(loss, accuracy))

    # save init summary
    with open(workspace + '/' + prefix + '/trained/{}_init_summary.txt'.format(network), 'w+') as f:
        with redirect_stdout(f):
            model.summary()

    # save model
    model.save(workspace + '/' + prefix + '/trained/{}_model'.format(network))

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
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help='Path to use for resuming training. Default is None so it begins training from 0. Path should be to savedModel file')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Model to use for training. Uses models from ../models/ directory')
    parser.add_argument('-n', '--network', type=str, default='custom',
                        help='Name of network. App uses this prop to save and load files')

    args = parser.parse_args()
    print('Command line options:')
    print(' --batch_size        : ', args.batch_size)
    print(' --epochs            : ', args.epochs)
    print(' --workspace         : ', args.workspace)
    print(' --prefix            : ', args.prefix)
    print(' --checkpoint        : ', args.checkpoint)
    print(' --model             : ', args.model)
    print(' --network           : ', args.network)

    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # create dir
    os.makedirs(args.workspace + '/' + args.prefix + '/trained', exist_ok=True)

    app(args.batch_size, args.epochs, args.workspace, args.prefix, args.checkpoint, args.model, args.network)
