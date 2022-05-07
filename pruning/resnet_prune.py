import tensorflow as tf
import tensorflow_datasets as tfds

def add_normalized_values(img, label):
    """Normalizes images"""
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return tf.image.resize(norm_img, [224, 224]), label


# load trained model
model = tf.keras.models.load_model('../models/resnet_50_trained.h5')

# dataset preparation
(ds_train, ds_validation) = tfds.load('imagenet2012', shuffle_files=True, as_supervised=True,
                                      split=['train', 'validation'])

ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_validation = ds_validation.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_train = ds_train.batch(100)
ds_validation = ds_validation.batch(100)


