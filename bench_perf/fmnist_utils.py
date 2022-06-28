import tensorflow as tf
import tensorflow_datasets as tfds

shape = (28, 28, 1)
output_classes = 10


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
