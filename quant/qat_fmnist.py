import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from contextlib import redirect_stdout


def add_normalized_values(img, label):
    """Normalizes images"""
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return norm_img, label


# load model
float_model = tf.keras.models.load_model('/workspace/vai_benchmark/data/models/pruned_fmnist')

# load dataset
(ds_train, ds_test) = tfds.load('fashion_mnist', split=['train', 'test'], as_supervised=True, shuffle_files=True)

# map data
ds_train = ds_train.batch(64)
ds_test = ds_test.batch(64)
ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# get qat model
quantizer = vitis_quantize.VitisQuantizer(float_model, quantize_strategy='8bit_tqt')
qat_model = quantizer.get_qat_model(init_quant=True, calib_dataset=ds_train)

# train
qat_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
float_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

qat_model.fit(ds_train, epochs=15)

# evaluate
float_res = float_model.evaluate(ds_test)
quant_res = qat_model.evaluate(ds_test)

# print results
print('Float model evaluation results', float_res)
print('Quantized model evaluation results', quant_res)

with open('/workspace/vai_benchmark/data/results/qat_fmnist.txt', 'w') as f:
    with redirect_stdout(f):
        print('Float model evaluation results', float_res)
        print('Quantized model evaluation results', quant_res)

# save model
qat_model.save('/workspace/vai_benchmark/data/models/quantized/quantized_qat_fmnist.h5')
