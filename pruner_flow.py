import tensorflow as tf
import tensorflow_datasets as tfds
from tf_nndct.optimization import IterativePruningRunner

input_shape = (28, 28, 1)
num_classes = 10

# load dataset
(ds_train, ds_test) = tfds.load('fashion_mnist', split=['train', 'test'], as_supervised=True, shuffle_files=True)

# create model
base_model = tf.keras.Sequential([
    tf.keras.Input(shape=input_shape),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(), tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
runner = IterativePruningRunner(base_model, input_spec)

ds_test = ds_test.batch(32)
ds_train = ds_train.batch(32)

def add_normalized_values(img, label):
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return norm_img, label


# map data
ds_test = ds_test.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)


# train
base_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
base_model.fit(ds_train, epochs=15)


def evaluate(model):
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    score = model.evaluate(ds_test, verbose=0)
    return score[1]


# evaluation
runner.ana(evaluate)

# prune
sparse_model = runner.prune(ratio=0.2)

sparse_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
sparse_model.fit(ds_train, batch_size=32, epochs=15, validation_split=0.1)
sparse_model.save_weights("model_sparse_0.2", save_format="tf")
