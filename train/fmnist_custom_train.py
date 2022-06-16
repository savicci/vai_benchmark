import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
from contextlib import redirect_stdout


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


def create_model(input_shape, num_classes) -> tf.keras.models.Model:
    return tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(), tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])


def app(batch_size, epochs, workspace, prefix):
    input_shape = (28, 28, 1)
    num_classes = 10

    # load dataset
    ds_train, ds_test = load_dataset(batch_size)

    # model to use
    model = create_model(input_shape, num_classes)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # train model
    model.fit(ds_train, epochs=epochs)

    # evaluate
    with open(workspace + '/' + prefix + '/trained/fmnist_evaluate.txt', 'w') as f:
        with redirect_stdout(f):
            loss, accuracy = model.evaluate(ds_test, verbose=2)
            print('Loss {}, accuracy {}'.format(loss, accuracy))

    # save init summary
    with open(workspace + '/' + prefix + '/trained/fmnist_init_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # save model
    model.save(workspace + '/' + prefix + '/trained/fmnist_model.h5')

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

    app(args.batch_size, args.epochs, args.workspace, args.prefix)
