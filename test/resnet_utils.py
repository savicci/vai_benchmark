from typing import List, Tuple
import tensorflow_datasets as tfds
import os

divider = '------------------------------------'


def load_tensorflow_dataset(size = None) -> Tuple[List, List]:
    data_dir = os.getenv('TFDS_DATA_DIR')
    ds_test = tfds.load('imagenet2012', split='validation', shuffle_files=True, data_dir=data_dir)

    images = []
    labels = []
    for record in tfds.as_numpy(ds_test):
        images.append(record['image'])
        labels.append(record['label'])

    if size is None:
        return images, labels
    else:
        return images[:size], labels[:size]


def preprocess_dataset(images, scale) -> List:
    return [record * (1 / 255.0) * scale for record in images]


def postprocess_results(out_vectors, labels):
    correct = 0
    miss = 0

    for i in range(len(out_vectors)):
        prediction = out_vectors[i]

        if prediction == labels[i]:
            correct += 1
        else:
            miss += 1

    accuracy = correct / len(out_vectors)
    print('Correct:%d, Wrong:%d, Accuracy:%.4f' % (correct, miss, accuracy))
    print(divider)
