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
    Norm = _get_norm_layer(norm)
    s = tf.keras.Input(shape=input_shape)
    c1 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (s)
    c1 = Norm()(c1)
    c1 = tf.nn.tanh(c1)
    c1 = tf.keras.layers.Dropout(0.5) (c1)
    c1 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (c1)
    c1 = Norm()(c1)
    c1 = tf.nn.tanh(c1)
    c1 = tf.keras.layers.Dropout(0.5) (c1)
    p1 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)) (c1)

    c2 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same') (p1)
    c2 = Norm()(c2)
    c2 = tf.nn.tanh(c2)
    c2 = tf.keras.layers.Dropout(0.5) (c2)
    c2 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same') (c2)
    c2 = Norm()(c2)
    c2 = tf.nn.tanh(c2)
    c2 = tf.keras.layers.Dropout(0.5) (c2)
    p2 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)) (c2)

    up1_2 = tf.keras.layers.Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), name='up12', padding='same')(c2)
    up1_2 = Norm()(up1_2)
    up1_2 = tf.nn.tanh(up1_2)
    conv1_2 = tf.keras.layers.concatenate([up1_2, c1], name='merge12', axis=3)
    c3 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (conv1_2)
    c3 = Norm()(c3)
    c3 = tf.nn.tanh(c3)
    c3 = tf.keras.layers.Dropout(0.5) (c3)
    c3 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (c3)
    c3 = Norm()(c3)
    c3 = tf.nn.tanh(c3)
    c3 = tf.keras.layers.Dropout(0.5) (c3)

    conv3_1 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same') (p2)
    conv3_1 = Norm()(conv3_1)
    conv3_1 = tf.nn.tanh(conv3_1)
    conv3_1 = tf.keras.layers.Dropout(0.5) (conv3_1)
    conv3_1 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same') (conv3_1)
    conv3_1 = Norm()(conv3_1)
    conv3_1 = tf.nn.tanh(conv3_1)
    conv3_1 = tf.keras.layers.Dropout(0.5) (conv3_1)
    pool3 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = tf.keras.layers.Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    up2_2 = Norm()(up2_2)
    up2_2 = tf.nn.tanh(up2_2)
    conv2_2 = tf.keras.layers.concatenate([up2_2, c2], name='merge22', axis=3) #x10
    conv2_2 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same') (conv2_2)
    conv2_2 = Norm()(conv2_2)
    conv2_2 = tf.nn.tanh(conv2_2)
    conv2_2 = tf.keras.layers.Dropout(0.5) (conv2_2)
    conv2_2 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same') (conv2_2)
    conv2_2 = Norm()(conv2_2)
    conv2_2 = tf.nn.tanh(conv2_2)
    conv2_2 = tf.keras.layers.Dropout(0.5) (conv2_2)

    up1_3 = tf.keras.layers.Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    up1_3 = Norm()(up1_3)
    up1_3 = tf.nn.tanh(up1_3)
    conv1_3 = tf.keras.layers.concatenate([up1_3, c1, c3], name='merge13', axis=3)

    conv1_3 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (conv1_3)
    conv1_3 = Norm()(conv1_3)
    conv1_3 = tf.nn.tanh(conv1_3)
    conv1_3 = tf.keras.layers.Dropout(0.5) (conv1_3)
    conv1_3 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (conv1_3)
    conv1_3 = Norm()(conv1_3)
    conv1_3 = tf.nn.tanh(conv1_3)
    conv1_3 = tf.keras.layers.Dropout(0.5) (conv1_3)

    conv4_1 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same') (pool3)
    conv4_1 = Norm()(conv4_1)
    conv4_1 = tf.nn.tanh(conv4_1)
    conv4_1 = tf.keras.layers.Dropout(0.5) (conv4_1)
    conv4_1 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same') (conv4_1)
    conv4_1 = Norm()(conv4_1)
    conv4_1 = tf.nn.tanh(conv4_1)
    conv4_1 = tf.keras.layers.Dropout(0.5) (conv4_1)
    pool4 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = tf.keras.layers.Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    up3_2 = Norm()(up3_2)
    up3_2 = tf.nn.tanh(up3_2)
    conv3_2 = tf.keras.layers.concatenate([up3_2, conv3_1], name='merge32', axis=3) #x20
    
    conv3_2 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same') (conv3_2)
    conv3_2 = Norm()(conv3_2)
    conv3_2 = tf.nn.tanh(conv3_2)
    conv3_2 = tf.keras.layers.Dropout(0.5) (conv3_2)
    conv3_2 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same') (conv3_2)
    conv3_2 = Norm()(conv3_2)
    conv3_2 = tf.nn.tanh(conv3_2)
    conv3_2 = tf.keras.layers.Dropout(0.5) (conv3_2)

    up2_3 = tf.keras.layers.Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    up2_3 = Norm()(up2_3)
    up2_3 = tf.nn.tanh(up2_3)
    conv2_3 = tf.keras.layers.concatenate([up2_3, c2, conv2_2], name='merge23', axis=3)
    conv2_3 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same') (conv2_3)
    conv2_3 = Norm()(conv2_3)
    conv2_3 = tf.nn.tanh(conv2_3)
    conv2_3 = tf.keras.layers.Dropout(0.5) (conv2_3)
    conv2_3 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same') (conv2_3)
    conv2_3 = Norm()(conv2_3)
    conv2_3 = tf.nn.tanh(conv2_3)
    conv2_3 = tf.keras.layers.Dropout(0.5) (conv2_3)

    up1_4 = tf.keras.layers.Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    up1_4 = Norm()(up1_4)
    up1_4 = tf.nn.tanh(up1_4)
    conv1_4 = tf.keras.layers.concatenate([up1_4, c1, c3, conv1_3], name='merge14', axis=3)
    conv1_4 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (conv1_4)
    conv1_4 = Norm()(conv1_4)
    conv1_4 = tf.nn.tanh(conv1_4)
    conv1_4 = tf.keras.layers.Dropout(0.5) (conv1_4)
    conv1_4 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (conv1_4)
    conv1_4 = Norm()(conv1_4)
    conv1_4 = tf.nn.tanh(conv1_4)
    conv1_4 = tf.keras.layers.Dropout(0.5) (conv1_4)

    conv5_1 = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same') (pool4)
    conv5_1 = Norm()(conv5_1)
    conv5_1 = tf.nn.tanh(conv5_1)
    conv5_1 = tf.keras.layers.Dropout(0.5) (conv5_1)
    conv5_1 = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same') (conv5_1)
    conv5_1 = Norm()(conv5_1)
    conv5_1 = tf.nn.tanh(conv5_1)
    conv5_1 = tf.keras.layers.Dropout(0.5) (conv5_1)

    up4_2 = tf.keras.layers.Conv2DTranspose(filters[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    up4_2 = Norm()(up4_2)
    up4_2 = tf.nn.tanh(up4_2)
    conv4_2 = tf.keras.layers.concatenate([up4_2, conv4_1], name='merge42', axis=3) #x30
    conv4_2 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same') (conv4_2)
    conv4_2 = Norm()(conv4_2)
    conv4_2 = tf.nn.tanh(conv4_2)
    conv4_2 = tf.keras.layers.Dropout(0.5) (conv4_2)
    conv4_2 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same') (conv4_2)
    conv4_2 = Norm()(conv4_2)
    conv4_2 = tf.nn.tanh(conv4_2)
    conv4_2 = tf.keras.layers.Dropout(0.5) (conv4_2)

    up3_3 = tf.keras.layers.Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    up3_3 = Norm()(up3_3)
    up3_3 = tf.nn.tanh(up3_3)
    conv3_3 = tf.keras.layers.concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
    conv3_3 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same') (conv3_3)
    conv3_3 = Norm()(conv3_3)
    conv3_3 = tf.nn.tanh(conv3_3)
    conv3_3 = tf.keras.layers.Dropout(0.5) (conv3_3)
    conv3_3 = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same') (conv3_3)
    conv3_3 = Norm()(conv3_3)
    conv3_3 = tf.nn.tanh(conv3_3)
    conv3_3 = tf.keras.layers.Dropout(0.5) (conv3_3)

    up2_4 = tf.keras.layers.Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    up2_4 = Norm()(up2_4)
    up2_4 = tf.nn.tanh(up2_4)
    conv2_4 = tf.keras.layers.concatenate([up2_4, c2, conv2_2, conv2_3], name='merge24', axis=3)
    conv2_4 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same') (conv2_4)
    conv2_4 = Norm()(conv2_4)
    conv2_4 = tf.nn.tanh(conv2_4)
    conv2_4 = tf.keras.layers.Dropout(0.5) (conv2_4)
    conv2_4 = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same') (conv2_4)
    conv2_4 = Norm()(conv2_4)
    conv2_4 = tf.nn.tanh(conv2_4)
    conv2_4 = tf.keras.layers.Dropout(0.5) (conv2_4)

    up1_5 = tf.keras.layers.Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    up1_5 = Norm()(up1_5)
    up1_5 = tf.nn.tanh(up1_5)
    conv1_5 = tf.keras.layers.concatenate([up1_5, c1, c3, conv1_3, conv1_4], name='merge15', axis=3)
    conv1_5 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (conv1_5)
    conv1_5 = Norm()(conv1_5)
    conv1_5 = tf.nn.tanh(conv1_5)
    conv1_5 = tf.keras.layers.Dropout(0.5) (conv1_5)
    conv1_5 = tf.keras.layers.Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same') (conv1_5)
    conv1_5 = Norm()(conv1_5)
    conv1_5 = tf.nn.tanh(conv1_5)
    conv1_5 = tf.keras.layers.Dropout(0.5) (conv1_5)

    nestnet_output_4 = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid', kernel_initializer = 'he_normal',  name='output_4', padding='same')(conv1_5)
    
    nestnet_output_4 = Norm()(nestnet_output_4)
    nestnet_output_4 = tf.nn.tanh(nestnet_output_4)
    return tf.keras.Model(inputs=s, outputs=[nestnet_output_4])


def AnotherUnetDiscriminator(input_shape=(256, 256, 3),
                             filters=[64, 128, 256, 512],
                             norm='instance_norm'):

    inputs = tf.keras.Input(shape=input_shape)
    Norm = _get_norm_layer(norm)
    h = inputs

    for filter_size in filters:
        h = tf.keras.layers.Conv2D(
            filter_size, (4, 4), strides=(2, 2), padding='same')(h)
        h = Norm()(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='same')(h)
    h = Norm()(h)
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
        activation='tanh',
        apply_initializer=True)  # (bs, 256, 256, 3)

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
        512, 4, strides=1, padding='valid', q=q, apply_initializer=True)(zero_pad1)  # (bs, 31, 31, 512)

    norm1 = Norm()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = Oper2D(
        1, 4, strides=1, padding='valid', q=q, apply_initializer=True)(zero_pad2)  # (bs, 30, 30, 1)

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
