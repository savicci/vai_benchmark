import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
from contextlib import redirect_stdout
from tf_nndct.optimization import IterativePruningRunner
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
    return tf.keras.models.load_model(workspace + '/' + prefix + '/trained/{}_model'.format(network))


def evaluate(model):
    """Function used by Pruner to evaluate pruning performance"""
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    score = model.evaluate(ds_test, verbose=0)
    return score[1]


def app(epochs, workspace, ratio, prefix, network):
    input_shape = (224, 224, 3)

    # model to use
    model = load_model(workspace, prefix, network)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # create prunning runner and analyze model
    input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
    pruning_runner = IterativePruningRunner(model, input_spec)
    pruning_runner.ana(evaluate)

    # prune
    sparse_model = pruning_runner.prune(ratio=ratio)

    # fine-tuning process
    sparse_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    sparse_model.fit(ds_train, epochs=epochs)

    # evaluate
    with open(workspace + '/' + prefix + '/pruned/{}_pruned_evaluate.txt'.format(network), 'w+') as f:
        with redirect_stdout(f):
            loss, accuracy = sparse_model.evaluate(ds_test, verbose=2)
            print('Loss {}, accuracy {}'.format(loss, accuracy))


    # save model
    filename = "/tmp/vai_benchmark/data/pruned/{}_model_sparse".format(network)
    sparse_model.save_weights(filename, save_format="tf")
    model.load_weights(filename)

    runner = IterativePruningRunner(model, input_spec)
    pruned_slim_model = runner.get_slim_model()
    pruned_slim_model.save(workspace + '/' + prefix + '/pruned/{}_model'.format(network))

    # save init summary
    with open(workspace + '/' + prefix + '/pruned/{}_pruned_summary.txt'.format(network), 'w+') as f:
        with redirect_stdout(f):
            pruned_slim_model.summary()

    print('Finished pruning and saving information')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--ratio', type=float, default='0.9',
                        help='Ratio to use for pruning. Float value from 0 to 1, 0.9 being default')
    parser.add_argument('-b', '--batch_size', type=int, default='32',
                        help='Batch size to use for training. Default is 32')
    parser.add_argument('-e', '--epochs', type=int, default='10',
                        help='Epoch number to finetune network. Default is 10')
    parser.add_argument('-w', '--workspace', type=str, default='/workspace/results',
                        help='Path to folder to write pruned model summary, evaluate and h5')
    parser.add_argument('-p', '--prefix', type=str, default='default',
                        help='Prefix to folder where all information will be written')
    parser.add_argument('-n', '--network', type=str, default='custom',
                        help='Name of network. App uses this prop to save and load files')

    args = parser.parse_args()
    print('Command line options:')
    print(' --ratio             : ', args.ratio)
    print(' --batch_size        : ', args.batch_size)
    print(' --epochs            : ', args.epochs)
    print(' --workspace         : ', args.workspace)
    print(' --prefix            : ', args.prefix)
    print(' --network           : ', args.network)

    # load dataset. Needs to be done earlier for evaluate function
    ds_train, ds_test = load_dataset(args.batch_size)

    # create dir
    os.makedirs(args.workspace + '/' + args.prefix + '/pruned', exist_ok=True)

    app(args.epochs, args.workspace, args.ratio, args.prefix, args.network)
