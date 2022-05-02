import tensorflow as tf
import tensorflow_datasets as tfds
from tf_nndct.optimization import IterativePruningRunner
from contextlib import redirect_stdout

# variables
SHARPNESS = 0.3
MAX_ITERATIONS = 15
DESIRED_ACCURACY = 0.85


def add_normalized_values(img, label):
    """Normalizes images"""
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return norm_img, label


def evaluate(model):
    """Function used by Pruner to evaluate pruning performance"""
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    score = model.evaluate(ds_test, verbose=0)
    return score[1]


def ratio_fun(x):
    """Calculates pruning ratio"""
    return x / (x + SHARPNESS)


def prune_loop(init_model):
    """Does all the work required to prune model"""
    input_spec = tf.TensorSpec((1, *input_shape), tf.float32)

    curr_accuracy = evaluate(init_model)
    if curr_accuracy <= DESIRED_ACCURACY:
        print("Will not start pruning. Accuracy below {}. Returning base model".format(DESIRED_ACCURACY))
        return init_model

    base_model = tf.keras.models.clone_model(init_model)

    for i in range(MAX_ITERATIONS):
        print("Iteration {}".format(i))

        # clone model in case that after pruning accuracy will be lower than desired
        prev_model = tf.keras.models.clone_model(base_model)

        pruning_runner = IterativePruningRunner(base_model, input_spec)
        pruning_runner.ana(evaluate)

        # pruning process
        ratio = ratio_fun(i)
        print("Using ratio {} for pruning".format(ratio))
        sparse_model = pruning_runner.prune(ratio=ratio)

        # fine-tuning process
        sparse_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        sparse_model.fit(ds_train, epochs=15)

        # accuracy evaluation process
        curr_accuracy = evaluate(sparse_model)
        if curr_accuracy <= DESIRED_ACCURACY:
            print("Finished pruning. Accuracy below {}. Returning previous model".format(DESIRED_ACCURACY))
            return prev_model
        else:
            print("Accuracy after pruning {}".format(curr_accuracy))

        # load sparse_model weights to base_model
        filename = "pruned/model_sparse_{}".format(i)
        sparse_model.save_weights(filename, save_format="tf")
        base_model.load_weights(filename)

    return base_model


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
init_model = tf.keras.Sequential([
    tf.keras.Input(shape=input_shape),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(), tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

# train
init_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
init_model.fit(ds_train, epochs=15)
init_model.evaluate(ds_test)

# save init summary
with open('init_model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        init_model.summary()

# pruning process
final_model = prune_loop(init_model)

print("Finished pruning")

# extracting pruned model (remove '0' weights)
spec = tf.TensorSpec((1, *input_shape), tf.float32)
runner = IterativePruningRunner(final_model, spec)
pruned_slim_model = runner.get_slim_model()

# save pruned summary
with open('pruned_model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        pruned_slim_model.summary()
