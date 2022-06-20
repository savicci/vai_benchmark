from typing import List, Tuple
import tensorflow_datasets as tfds
import tensorflow as tf
import os

divider = '------------------------------------'


def load_tensorflow_dataset(size=None) -> Tuple[List, List]:
    data_dir = os.getenv('TFDS_DATA_DIR')
    ds_test = tfds.load('imagenet2012', split='validation', shuffle_files=True, data_dir=data_dir)

    images = []
    labels = []
    i = 0
    for record in tfds.as_numpy(ds_test):
        images.append(tf.image.resize(record['image'], [224, 224]))
        labels.append(record['label'])

        i += 1
        if i == size:
            break

    return images, labels


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
