import tensorflow as tf
import fmnist_utils


def shortcut(channel_in, channel_out):
    if channel_in == channel_out:
        return lambda x: x
    else:
        return projection(channel_out)


def projection(channel_out):
    return tf.keras.layers.Conv2D(channel_out, kernel_size=(1, 1), padding="same")


def residual_block(channel_in=64, channel_out=256):
    channel = channel_out // 4

    layers = [
        tf.keras.layers.Conv2D(channel, kernel_size=(1, 1), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(tf.nn.relu),
        tf.keras.layers.Conv2D(channel, kernel_size=(3, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(tf.nn.relu),
        tf.keras.layers.Conv2D(channel_out, kernel_size=(3, 3), padding="same"),
        tf.keras.layers.BatchNormalization()
    ]

    if channel_in != channel_out:
        layers.append(tf.keras.layers.Conv2D(channel_out, kernel_size=(1, 1), padding="same"))

    layers.append(tf.keras.layers.Activation(tf.nn.relu))

    return layers


def customized_resnet(input_shape, output_dim):
    layers = [
        # conv1
        tf.keras.layers.Conv2D(64, input_shape=input_shape, kernel_size=(7, 7), strides=(2, 2), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(tf.nn.relu),
        # conv2_x
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
    ]
    layers.extend(residual_block(64, 256))

    for _ in range(2):
        layers.extend(residual_block(256, 256))

    layers.append(tf.keras.layers.Conv2D(512, kernel_size=(1, 1), strides=(2, 2)))

    for _ in range(2):
        layers.extend(residual_block(512, 512))

    layers.append(tf.keras.layers.Conv2D(1024, kernel_size=(1, 1), strides=(2, 2)))
    for _ in range(2):
        layers.extend(residual_block(1024, 1024))

    # layers.append(tf.keras.layers.Conv2D(2048, kernel_size=(1, 1), strides=(2, 2)))
    # for _ in range(2):
    #     layers.extend(residual_block(2048, 2048))

    layers.extend([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1000, activation=tf.nn.relu),
        tf.keras.layers.Dense(output_dim, activation=tf.nn.softmax)
    ])

    return tf.keras.models.Sequential(layers)


if __name__ == '__main__':
    batch_size = 128

    ds_train, ds_test = fmnist_utils.load_dataset(128)

    model = customized_resnet((28, 28, 1), 10)

    model.summary()

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(ds_train)

    from tensorflow_model_optimization.quantization.keras import vitis_quantize

    quantizer = vitis_quantize.VitisQuantizer(model)
    quantized_model = quantizer.quantize_model(calib_dataset=ds_train, calib_steps=10)
    quantized_model.save('./fmnist_custom_resnet_model.h5')
