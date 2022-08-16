import bench_perf.resnet_seq as resnet_seq
import bench_perf.fmnist_utils as fmnist_utils
import argparse
import os
from contextlib import redirect_stdout

input_shape = (28, 28, 1)
output_shape = 10


def app(batch_size, epochs, path):
    ds_train, ds_test = fmnist_utils.load_dataset(batch_size)

    model = resnet_seq.default_resnet(input_shape, output_shape)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.summary()

    model.fit(ds_train, validation_data=ds_test, epochs=epochs)

    with open('{}/init_trained_evaluate.txt'.format(path), 'w') as f:
        with redirect_stdout(f):
            model.evaluate(ds_test)

    with open('{}/init_trained_summary.txt'.format(path), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    model.save('{}/init_trained_model.h5'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default='128',
                        help='Batch size to use for training. Default is 128')
    parser.add_argument('-e', '--epochs', type=int, default='20',
                        help='Epoch number to train network. Default is 20')
    parser.add_argument('-p', '--path', type=str, default='/workspace/vai_benchmark/data/resnet_fmnist/trained',
                        help='Path to folder where all information will be written. Default /workspace/vai_benchmark/data/resnet_fmnist/trained')

    args = parser.parse_args()
    print('Command line options:')
    print(' --batch_size   : ', args.batch_size)
    print(' --epochs   : ', args.epochs)
    print(' --path   : ', args.path)

    # create dir
    os.makedirs(args.path, exist_ok=True)

    app(args.batch_size, args.epochs, args.path)
