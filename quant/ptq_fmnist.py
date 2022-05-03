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
(ds_train, ds_test) = tfds.load('fashion_mnist', split=['train', 'test'], as_supervised=True, shuffle_files=True)

# map data
ds_train = ds_train.batch(10)
ds_test = ds_test.batch(64)
ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# create quantizer
quantizer = vitis_quantize.VitisQuantizer(float_model)

# quantize with fine tuning
quantized_model = quantizer.quantize_model(calib_dataset=ds_train, calib_steps=100, calib_batch_size=10)

# save
quantized_model.save('/workspace/vai_benchmark/data/models/quantized/quantized_fmnist.h5')

float_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
float_res = float_model.evaluate(ds_test)

quantized_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
quant_res = quantized_model.evaluate(ds_test)

print('Float model evaluation results', float_res)
print('Quantized model evaluation results', quant_res)

with open('/workspace/vai_benchmark/data/results/ptq_fmnist.txt', 'w') as f:
    f.write("Float model results")
    f.write(float_res)
    f.write('\n')
    f.write("Quantized model results")
    f.write(quant_res)