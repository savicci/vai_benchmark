import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, \
    Add
from tensorflow.keras import Model
import fmnist_utils
from tensorflow_model_optimization.quantization.keras import vitis_quantize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Path to trained h5 model. Default is None and starts training.')

args = parser.parse_args()


class ResidualBlock(Model):
    def __init__(self, channel_in=64, channel_out=256):
        super().__init__()

        channel = channel_out // 4

        self.conv1 = Conv2D(channel, kernel_size=(1, 1), padding="same")
        self.bn1 = BatchNormalization()
        self.av1 = Activation(tf.nn.relu)
        self.conv2 = Conv2D(channel, kernel_size=(3, 3), padding="same")
        self.bn2 = BatchNormalization()
        self.av2 = Activation(tf.nn.relu)
        self.conv3 = Conv2D(channel_out, kernel_size=(1, 1), padding="same")
        self.bn3 = BatchNormalization()
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.add = Add()
        self.av3 = Activation(tf.nn.relu)

    def call(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.av1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.av2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        h = self.add([h, shortcut])
        y = self.av3(h)
        return y

    def _shortcut(self, channel_in, channel_out):
        if channel_in == channel_out:
            return lambda x: x
        else:
            return self._projection(channel_out)

    def _projection(self, channel_out):
        return Conv2D(channel_out, kernel_size=(1, 1), padding="same")


class ResNet50(Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()

        self._layers = [
            # conv1
            Conv2D(64, input_shape=input_shape, kernel_size=(7, 7), strides=(2, 2), padding="same"),
            BatchNormalization(),
            Activation(tf.nn.relu),
            # conv2_x
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            ResidualBlock(64, 256),
            [
                ResidualBlock(256, 256) for _ in range(2)
            ],
            # conv3_x
            Conv2D(512, kernel_size=(1, 1), strides=(2, 2)),
            [
                ResidualBlock(512, 512) for _ in range(4)
            ],
            # conv4_x
            Conv2D(1024, kernel_size=(1, 1), strides=(2, 2)),
            [
                ResidualBlock(1024, 1024) for _ in range(6)
            ],
            # conv5_x
            Conv2D(2048, kernel_size=(1, 1), strides=(2, 2)),
            [
                ResidualBlock(2048, 2048) for _ in range(3)
            ],
            # last part
            GlobalAveragePooling2D(),
            Dense(1000, activation=tf.nn.relu),
            Dense(output_dim, activation=tf.nn.softmax)
        ]

    def call(self, x):
        for layer in self._layers:
            if isinstance(layer, list):
                for l in layer:
                    x = l(x)
            else:
                x = layer(x)
        return x


if args.checkpoint is None:
    model = ResNet50((28, 28, 1), 10)
    model.build(input_shape=(None, 28, 28, 1))
else:
    model = tf.keras.models.load_model(args.checkpoint)

model.summary()

batch_size = 128

ds_train, ds_test = fmnist_utils.load_dataset(batch_size)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

if args.checkpoint is None:
    model.fit(ds_train, epochs=1)

model.save('./fmnist_trained_ckpt.h5')

quantizer = vitis_quantize.VitisQuantizer(model)
quantized_model = quantizer.quantize_model(calib_dataset=ds_train, calib_steps=10)

quantized_model.save('./fmnist_resnet_model.h5')
