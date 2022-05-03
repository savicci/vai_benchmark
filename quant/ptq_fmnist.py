import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_model_optimization.quantization.keras import vitis_quantize

def add_normalized_values(img, label):
    """Normalizes images"""
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return norm_img, label


# load model
float_model = tf.keras.models.load_model('/workspace/vai_benchmark/data/models/pruned_fmnist')

# load calibration dataset
ds_train = tfds.load('fashion_mnist', split='train', as_supervised=True, shuffle_files=True)

# map data
ds_train = ds_train.batch(32)
ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# create quantizer
quantizer = vitis_quantize.VitisQuantizer(float_model)

# quantize with fine tuning
quantized_model = quantizer.quantize_model(calib_dataset=ds_train, calib_batch_size=32, include_fast_ft=True, fast_ft_epochs=5)

# save
quantized_model.save('/workspace/vai_benchmark/data/models/quantized_fmnist.h5')