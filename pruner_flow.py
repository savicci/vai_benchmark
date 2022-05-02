import tensorflow as tf
import tensorflow_datasets as tfds
from tf_nndct.optimization import IterativePruningRunner


def add_normalized_values(img, label):
    '''Normalizes images'''
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return norm_img, label


def evaluate(model):
    '''Function used by Pruner to evaluate pruning performance'''
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    score = model.evaluate(ds_test, verbose=0)
    return score[1]


def pruning_loop(model):
    ratio = 0.5
    iteration = 0

    input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
    curr_accuracy = evaluate(model)

    while curr_accuracy >= 0.85:
        print("Iteration {}".format(iteration))
        print("Accuracy {}".format(curr_accuracy))

        # setup
        filename = "pruned/model_sparse_{}".format(iteration)
        pruning_runner = IterativePruningRunner(model, input_spec)
        pruning_runner.ana(evaluate)

        # run pruning
        sparse_model = pruning_runner.prune(ratio=ratio)
        sparse_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        sparse_model.fit(ds_train, epochs=15)
        sparse_model.save_weights(filename, save_format="tf")

        # post pruning setup for next iteration
        iteration += 1
        curr_accuracy = evaluate(sparse_model)
        model.load_weights(filename)

        # decrease pruning ratio
        ratio *= 0.8

    return model


input_shape = (28, 28, 1)
num_classes = 10

# load dataset
(ds_train, ds_test) = tfds.load('fashion_mnist', split=['train', 'test'], as_supervised=True, shuffle_files=True)

# map data
ds_test = ds_test.batch(32)
ds_train = ds_train.batch(32)

ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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

# train
base_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
base_model.fit(ds_train, epochs=15)
base_model.evaluate(ds_test)

base_model.summary()

pruned_model = pruning_loop(base_model)

spec = tf.TensorSpec((1, *input_shape), tf.float32)
runner = IterativePruningRunner(pruned_model, spec)
pruned_slim_model = runner.get_slim_model()

pruned_slim_model.summary()
