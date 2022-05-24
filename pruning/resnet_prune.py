import tensorflow as tf
import tensorflow_datasets as tfds
from tf_nndct.optimization import IterativePruningRunner
import argparse

input_shape = (224, 224, 3)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--ratio', default=0.5, help='Ratio to prune. Float value from 0 to 1', type=float)
args = parser.parse_args()


def add_normalized_values(img, label):
    """Normalizes images"""
    norm_img = tf.cast(img, dtype=tf.float32) / 255.0
    return tf.image.resize(norm_img, [224, 224]), label


def evaluate(model):
    """Function used by Pruner to evaluate pruning performance"""
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    score = model.evaluate(ds_validation, verbose=0)
    return score[1]


# load trained model
trained_model = tf.keras.models.load_model('/workspace/vai_benchmark/models/resnet_50_trained.h5')

# dataset preparation
(ds_train, ds_validation) = tfds.load('imagenet2012', shuffle_files=True, as_supervised=True,
                                      split=['train', 'validation'])

ds_train = ds_train.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_validation = ds_validation.map(add_normalized_values, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_train = ds_train.batch(64)
ds_validation = ds_validation.batch(64)

input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
pruning_runner = IterativePruningRunner(trained_model, input_spec)

pruning_runner.ana(evaluate)

pruned_model = pruning_runner.prune(ratio=args.ratio)

print('Evaluation after pruning')
pruned_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(pruned_model.evaluate(ds_validation, verbose=0))

print("Fine tuning")
pruned_model.fit(ds_train, validation_data=ds_validation, epochs=5)

print('Evaluation after fine tuning')
print(pruned_model.evaluate(ds_validation, verbose=0))

spec = tf.TensorSpec((1, *input_shape), tf.float32)
runner = IterativePruningRunner(pruned_model, spec)
pruned_slim_model = runner.get_slim_model()

pruned_slim_model.save('/workspace/vai_benchmark/data/models/pruned_trained_resnet')



