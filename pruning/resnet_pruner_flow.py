import tensorflow as tf
import tensorflow_datasets as tfds
from tf_nndct.optimization import IterativePruningRunner
from contextlib import redirect_stdout
import argparse

tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', default=None, help='path to checkpoint to use', type=str)
parser.add_argument('-s', '--skip_train', default=False,
                    help='Indicates whether to skip training and go straight to pruning', type=bool)
args = parser.parse_args()

# variables
SHARPNESS = 0.3
MAX_ITERATIONS = 8
DESIRED_ACCURACY = 0.75


def add_normalized_values(img, label):
    """Normalizes images"""
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return norm_img, label


def evaluate(model):
    """Function used by Pruner to evaluate pruning performance"""
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    score = model.evaluate(ds_validation, verbose=0)
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

    for i in range(1, MAX_ITERATIONS):
        print("Iteration {}".format(i))

        # clone model in case that after pruning accuracy will be lower than desired
        prev_model = tf.keras.models.clone_model(base_model)

        pruning_runner = IterativePruningRunner(base_model, input_spec)

        # will analyze once, then use cached results
        pruning_runner.ana(evaluate)

        # pruning process
        ratio = ratio_fun(i)
        print("Using ratio {} for pruning".format(ratio))
        sparse_model = pruning_runner.prune(ratio=ratio)

        # fine-tuning process
        sparse_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        sparse_model.fit(ds_train, epochs=10)

        # accuracy evaluation process
        curr_accuracy = evaluate(sparse_model)
        if curr_accuracy <= DESIRED_ACCURACY:
            print("Finished pruning. Accuracy below {}. Returning previous model".format(DESIRED_ACCURACY))
            return prev_model
        else:
            print("Accuracy after pruning {}".format(curr_accuracy))

        # load sparse_model weights to base_model
        filename = "/workspace/vai_benchmark/data/pruned/resnet_model_sparse_{}".format(i)
        sparse_model.save_weights(filename, save_format="tf")
        base_model.load_weights(filename)

    return base_model


# load dataset
(ds_train, ds_validation) = tfds.load('imagenet_resized/64x64', split=['train', 'validation'], as_supervised=True,
                                      shuffle_files=True)

# map data
ds_validation = ds_validation.batch(192)
ds_train = ds_train.batch(192)

ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_validation = ds_validation.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# create model
input_shape = (64, 64, 3)

if args.checkpoint is not None:
    init_model = tf.keras.models.load_model(args.checkpoint)
else:
    input_t = tf.keras.Input(shape=input_shape)
    res_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, input_tensor=input_t)

    init_model = tf.keras.models.Sequential()
    init_model.add(res_model)
    init_model.add(tf.keras.layers.Flatten())
    init_model.add(tf.keras.layers.Dense(1000, activation='softmax'))

# train
init_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

if args.skip_train is not True:
    init_model.fit(ds_train, epochs=15)

init_model.evaluate(ds_validation)

# save just in case we need more training
init_model.save('/workspace/vai_benchmark/models/resnet50_15e')

# save init summary
with open('/workspace/vai_benchmark/data/results/resnet_init_model_summary.txt', 'w') as f:
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
with open('/workspace/vai_benchmark/data/results/resnet_pruned_model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        pruned_slim_model.summary()

pruned_slim_model.save('/workspace/vai_benchmark/data/models/pruned_resnet')
