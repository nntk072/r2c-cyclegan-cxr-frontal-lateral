import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from layers import Oper2D, Oper2DTranspose
from layers import downsample, upsample, integrate_oper_upsample, integrate_oper_downsample

# ==============================================================================
# =                                  networks                                  =
# ==============================================================================


def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def ResnetGenerator(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)

        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(
            dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = keras.layers.Conv2DTranspose(
            dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=inputs, outputs=h)


def ConvDiscriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = keras.layers.Conv2D(
            dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = keras.layers.Conv2D(
        dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    return keras.Model(inputs=inputs, outputs=h)


"""
def OpGenerator(input_shape=(256, 256, 3), q=1):

    dim = 64
    Norm = tfa.layers.InstanceNormalization

    def _residual_block(x):
        dim = x.shape[-1]
        x1 = tf.nn.tanh(x)
        h = x1

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = Oper2D(dim, 3, q=q, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.tanh(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = Oper2D(dim, 3, q=q, padding='valid', use_bias=False)(h)
        h = Norm()(h)

        return tf.nn.tanh(tf.keras.layers.add([x, h]))

    h = inputs = tf.keras.Input(shape=input_shape)

    dim *= 2
    h = Oper2D(dim, 3, strides=2, q=q, padding='same', use_bias=False)(h)
    h = Norm()(h)

    h = _residual_block(h)

    # # Classification branch.
    # x = tf.keras.layers.MaxPool2D(h.shape[1])(h)
    # x = tf.keras.layers.Flatten()(x)
    # y_class = tf.keras.layers.Dense(2, activation='softmax')(x)

    dim //= 2
    h = Oper2DTranspose(3, 3, strides=2, q=q, padding='same')(h)
    h = tf.nn.tanh(h)

    # return tf.keras.Model(inputs=inputs, outputs=[h, y_class])
    return tf.keras.Model(inputs=inputs, outputs=h)

def OpDiscriminator(input_shape=(256, 256, 3), q=1):
    dim = 64
    Norm = tfa.layers.InstanceNormalization

    h = inputs = tf.keras.Input(shape=input_shape)

    h = Oper2D(dim, 4, strides=2, q=q, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    h = Oper2D(2 * dim, 4, strides=4, q=q, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    h = Oper2D(1, 4, strides=1, q=q, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    return tf.keras.Model(inputs=inputs, outputs=h)
"""


def OpGenerator(input_shape=(256, 256, 3),
                output_channels=3,
                dim=64,
                n_downsamplings=2,
                n_blocks=3,
                norm='instance_norm',
                q=1):
    Norm = _get_norm_layer(norm)

    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = Oper2D(dim, 3, q=q, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.tanh(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = Oper2D(dim, 3, q=q, padding='valid', use_bias=False)(h)
        h = Norm()(h)

        return tf.nn.tanh(keras.layers.add([x, h]))

    h = inputs = keras.Input(shape=input_shape)

    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = Oper2D(dim, 7, q=q, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.tanh(h)

    for _ in range(n_downsamplings):
        dim *= 2
        h = Oper2D(dim, 3, strides=2, q=q, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.tanh(h)

    for _ in range(n_blocks):
        h = _residual_block(h)

    for _ in range(n_downsamplings):
        dim //= 2
        h = Oper2DTranspose(dim, 3, strides=2, q=q,
                            padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.tanh(h)

    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = Oper2D(output_channels, 7, q=q, padding='valid')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=inputs, outputs=h)


def OpDiscriminator(input_shape=(256, 256, 3),
                    dim=64,
                    n_downsamplings=3,
                    norm='instance_norm',
                    q=1):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    h = inputs = keras.Input(shape=input_shape)

    h = Oper2D(dim, 4, strides=2, q=q, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = Oper2D(dim, 4, strides=2, q=q, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    dim = min(dim * 2, dim_ * 8)
    h = Oper2D(dim, 4, strides=1, q=q, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    h = Oper2D(1, 4, strides=1, q=q, padding='same')(h)

    return keras.Model(inputs=inputs, outputs=h)


def UNetGenerator(input_shape=(256, 256, 3),
                  output_channels=3,
                  norm='instance_norm'):
    """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

    Args:
      output_channels: Output channels
      norm: Type of normalization. Either 'batch_norm' or 'instance_norm'.

    Returns:
      Generator model
    """
    Norm = _get_norm_layer(norm)
    down_stack = [
        downsample(64, 4, norm, apply_norm=False),  # (bs, 128, 128, 64)
        downsample(128, 4, norm),  # (bs, 64, 64, 128)
        downsample(256, 4, norm),  # (bs, 32, 32, 256)
        downsample(512, 4, norm),  # (bs, 16, 16, 512)
        downsample(512, 4, norm),  # (bs, 8, 8, 512)
        downsample(512, 4, norm),  # (bs, 4, 4, 512)
        downsample(512, 4, norm),  # (bs, 2, 2, 512)
        downsample(512, 4, norm),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, norm, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, norm, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, norm, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4, norm),  # (bs, 16, 16, 1024)
        upsample(256, 4, norm),  # (bs, 32, 32, 512)
        upsample(128, 4, norm),  # (bs, 64, 64, 256)
        upsample(64, 4, norm),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')  # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def UNetDiscriminator(input_shape=(256, 256, 3),
                      norm='instance_norm',
                      target=False):
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

    Args:
      norm: Type of normalization. Either 'batch_norm' or 'instance_norm'.
      target: Bool, indicating whether target image is an input or not.

    Returns:
      Discriminator model
    """
    Norm = _get_norm_layer(norm)
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
        x = tf.keras.layers.concatenate(
            [inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, norm, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4, norm)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4, norm)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    norm1 = Norm()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)


def AnotherUNetGenerator(input_shape=(256, 256, 3),
                         num_classes=3,
                         filters=[64, 128, 256, 512],
                         dropout_rate=0.5,
                         activation='elu',
                         padding='same',
                         norm='instance_norm'
                         ):

    inputs = tf.keras.Input(shape=input_shape)
    Norm = _get_norm_layer(norm)
    conv_blocks = []
    pool_layers = []

    h = inputs

    # Downward path
    for filter_size in filters[:-1]:
        if filter_size == 64:
            h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
            h = keras.layers.Conv2D(
                filter_size, 7, padding='valid', use_bias=False)(h)
            h = Norm()(h)
            h = tf.nn.relu(h)
            h = tf.keras.layers.Dropout(dropout_rate)(h)
        elif filter_size == 256:
            h = keras.layers.Conv2D(filter_size, (3, 3), padding=padding)(h)
            h = Norm()(h)
            h = tf.nn.relu(h)
            h = tf.keras.layers.Dropout(dropout_rate)(h)
            h = keras.layers.Conv2D(filter_size, (3, 3), padding=padding)(h)
            h = Norm()(h)
            h = tf.nn.relu(h)
            h = tf.keras.layers.Dropout(dropout_rate)(h)

            x = h
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = keras.layers.Conv2D(
                filter_size, 3, padding='valid', use_bias=False)(x)
            x = Norm()(x)
            x = tf.nn.relu(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)

            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = keras.layers.Conv2D(
                filter_size, 3, padding='valid', use_bias=False)(x)
            x = Norm()(x)
            x = tf.nn.relu(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            h = keras.layers.add([h, x])

            x = h
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = keras.layers.Conv2D(
                filter_size, 3, padding='valid', use_bias=False)(x)
            x = Norm()(x)
            x = tf.nn.relu(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)

            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = keras.layers.Conv2D(
                filter_size, 3, padding='valid', use_bias=False)(x)
            x = Norm()(x)
            x = tf.nn.relu(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            h = keras.layers.add([h, x])

            x = h
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = keras.layers.Conv2D(
                filter_size, 3, padding='valid', use_bias=False)(x)
            x = Norm()(x)
            x = tf.nn.relu(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)

            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = keras.layers.Conv2D(
                filter_size, 3, padding='valid', use_bias=False)(x)
            x = Norm()(x)
            x = tf.nn.relu(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            h = keras.layers.add([h, x])

        else:
            h = keras.layers.Conv2D(filter_size, (3, 3), padding=padding)(h)
            h = Norm()(h)
            h = tf.nn.relu(h)
            h = tf.keras.layers.Dropout(dropout_rate)(h)
            h = keras.layers.Conv2D(filter_size, (3, 3), padding=padding)(h)
            h = Norm()(h)
            h = tf.nn.relu(h)
            h = tf.keras.layers.Dropout(dropout_rate)(h)
        conv_blocks.append(h)
        pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(h)
        pool_layers.append(pool)
        h = pool

    # Bottom
    h = keras.layers.Conv2D(filters[-1], (3, 3),
                            padding=padding)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)
    h = tf.keras.layers.Dropout(dropout_rate)(h)
    h = keras.layers.Conv2D(filters[-1], (3, 3),
                            padding=padding)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)
    h = tf.keras.layers.Dropout(dropout_rate)(h)

    # Upward path
    for i, filter_size in enumerate(filters[:-1][::-1]):
        h = keras.layers.Conv2DTranspose(filter_size, (2, 2),
                                         strides=(2, 2), padding=padding)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)
        h = keras.layers.Concatenate(axis=3)([h, conv_blocks[-(i+1)]])
        h = keras.layers.Conv2D(filter_size, (3, 3), padding=padding)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)
        h = tf.keras.layers.Dropout(dropout_rate)(h)
        h = keras.layers.Conv2D(filter_size, (3, 3), padding=padding)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)
        h = tf.keras.layers.Dropout(dropout_rate)(h)

    # Output layer
    # outputs = keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid',
    #                               padding=padding)(h)
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(num_classes, 7, padding='valid')(h)
    h = tf.tanh(h)
    model = tf.keras.Model(inputs=inputs, outputs=h)

    return model


def AnotherUnetDiscriminator(input_shape=(256, 256, 3),
                             filters=[64, 128, 256, 512]):

    inputs = tf.keras.Input(shape=input_shape)
    h = inputs

    for filter_size in filters:
        h = tf.keras.layers.Conv2D(
            filter_size, (4, 4), strides=(2, 2), padding='same')(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='same')(h)
    outputs = tf.keras.activations.sigmoid(h)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def OpUNetGenerator(input_shape=(256, 256, 3),
                    output_channels=3,
                    norm='instance_norm',
                    q=1):
    """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

    Args:
      output_channels: Output channels
      norm: Type of normalization. Either 'batch_norm' or 'instance_norm'.
      q: number of convolutional layers in Oper2D and Oper2DTranspose

    Returns:
      Generator model
    """
    down_stack = [
        integrate_oper_downsample(
            64, 4, norm, apply_norm=False, q=q),  # (bs, 128, 128, 64)
        integrate_oper_downsample(128, 4, norm, q=q),  # (bs, 64, 64, 128)
        integrate_oper_downsample(256, 4, norm, q=q),  # (bs, 32, 32, 256)
        integrate_oper_downsample(512, 4, norm, q=q),  # (bs, 16, 16, 512)
        integrate_oper_downsample(512, 4, norm, q=q),  # (bs, 8, 8, 512)
        integrate_oper_downsample(512, 4, norm, q=q),  # (bs, 4, 4, 512)
        integrate_oper_downsample(512, 4, norm, q=q),  # (bs, 2, 2, 512)
        integrate_oper_downsample(512, 4, norm, q=q),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        integrate_oper_upsample(
            512, 4, norm, apply_dropout=True, q=q),  # (bs, 2, 2, 1024)
        integrate_oper_upsample(
            512, 4, norm, apply_dropout=True, q=q),  # (bs, 4, 4, 1024)
        integrate_oper_upsample(
            512, 4, norm, apply_dropout=True, q=q),  # (bs, 8, 8, 1024)
        integrate_oper_upsample(512, 4, norm, q=q),  # (bs, 16, 16, 1024)
        integrate_oper_upsample(256, 4, norm, q=q),  # (bs, 32, 32, 512)
        integrate_oper_upsample(128, 4, norm, q=q),  # (bs, 64, 64, 256)
        integrate_oper_upsample(64, 4, norm, q=q),  # (bs, 128, 128, 128)
    ]

    last = Oper2DTranspose(
        output_channels, 4, strides=2,
        padding='same', q=q,
        activation='tanh')  # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def OpUNetDiscriminator(input_shape=(256, 256, 3),
                        norm='instance_norm',
                        target=False,
                        q=1):
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

    Args:
      norm: Type of normalization. Either 'batch_norm' or 'instance_norm'.
      target: Bool, indicating whether target image is an input or not.
      q: number of convolutional layers in Oper2D

    Returns:
      Discriminator model
    """
    Norm = _get_norm_layer(norm)
    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
        x = tf.keras.layers.concatenate(
            [inp, tar])  # (bs, 256, 256, channels*2)

    down1 = integrate_oper_downsample(
        64, 4, norm, False, q=q)(x)  # (bs, 128, 128, 64)
    down2 = integrate_oper_downsample(
        128, 4, norm, q=q)(down1)  # (bs, 64, 64, 128)
    down3 = integrate_oper_downsample(
        256, 4, norm, q=q)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = Oper2D(
        512, 4, strides=1, padding='valid', q=q)(zero_pad1)  # (bs, 31, 31, 512)

    norm1 = Norm()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = Oper2D(
        1, 4, strides=1, padding='valid', q=q)(zero_pad2)  # (bs, 30, 30, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)
# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================


class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(
            initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate *
            (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
