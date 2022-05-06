import tensorflow as tf
import tensorflow_datasets as tfds

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)

def add_normalized_values(img, label):
    """Normalizes images"""
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return tf.image.resize(norm_img, [224, 224]), label


# dataset preparation
(ds_train, ds_validation) = tfds.load('imagenet2012', shuffle_files=True, as_supervised=True, split=['train', 'validation'])

ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_validation = ds_validation.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_train = ds_train.batch(192)
ds_validation = ds_validation.batch(192)

# load model
float_model = tf.keras.models.load_model('../models/resnet_50.h5')

float_model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

float_model.fit(ds_train, epochs=150)

float_model.save('../models/resnet_50_trained.h5')

