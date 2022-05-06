import tensorflow as tf
import tensorflow_datasets as tfds

def add_normalized_values(img, label):
    """Normalizes images"""
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    norm_img = tf.image.resize(norm_img, [224, 224])
    return norm_img, label


# dataset preparation
(ds_train, ds_validation) = tfds.load('imagenet2012', shuffle_files=True, as_supervised=True, split=['train', 'validation'])

ds_train = ds_train.batch(192)
ds_validation = ds_validation.batch(192)

ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_validation = ds_validation.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)


# load model
float_model = tf.keras.models.load_model('../models/resnet_50.h5')

float_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics='accuracy'
)

float_model.evaluate(ds_validation, verbose=2)



