import tensorflow as tf
import tensorflow_datasets as tfds
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', default=None, help='path to checkpoint to use', type=str)
args = parser.parse_args()

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

def add_normalized_values(img, label):
    """Normalizes images"""
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return tf.image.resize(norm_img, [224, 224]), label


# dataset preparation
(ds_train, ds_validation) = tfds.load('imagenet2012', shuffle_files=True, as_supervised=True,
                                      split=['train', 'validation'])

ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_validation = ds_validation.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_train = ds_train.batch(64)
ds_validation = ds_validation.batch(64)

# load model from vitis ai pre optimized models
if args.checkpoint is not None:
    float_model = tf.keras.models.load_model(args.checkpoint)
else:
    float_model = tf.keras.models.load_model('../models/resnet_50.h5')

float_model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

checkpoint_path = '/workspace/vai_benchmark/data/train/checkpoints/resnet.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False, verbose=1, period=5)

float_model.fit(ds_train, validation_data=ds_validation, epochs=30, callbacks=[cp_callback])

float_model.save('../models/resnet_50_trained.h5')
