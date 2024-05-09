import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

####################################################
# Operational Layers.


class Oper2D(tf.keras.Model):
    def __init__(self, filters, kernel_size, activation=None, q=1, padding='valid', use_bias=True, strides=1, apply_initializer=False):
        super(Oper2D, self).__init__(name='')

        self.activation = activation
        self.q = q
        self.all_layers = []
        initializer = tf.random_normal_initializer(0., 0.02)
        for i in range(0, q):  # q convolutional layers.
            if not apply_initializer:
                self.all_layers.append(tf.keras.layers.Conv2D(filters,
                                                              (kernel_size,
                                                               kernel_size),
                                                              padding=padding,
                                                              use_bias=use_bias,
                                                              strides=strides,
                                                              activation=activation))
            else:
                self.all_layers.append(tf.keras.layers.Conv2D(filters,
                                                              (kernel_size,
                                                               kernel_size),
                                                              padding=padding,
                                                              use_bias=use_bias,
                                                              strides=strides,
                                                              activation=activation,
                                                              kernel_initializer=initializer))

    @tf.function
    def call(self, input_tensor, training=False):

        x = self.all_layers[0](input_tensor)  # First convolutional layer.

        if self.q > 1:
            for i in range(1, self.q):
                x += self.all_layers[i](tf.math.pow(input_tensor, i + 1))

        if self.activation is not None:
            return eval('tf.nn.' + self.activation + '(x)')
        else:
            return x

####################################################
# Transposed Operational Layers.


class Oper2DTranspose(tf.keras.Model):
    def __init__(self, filters, kernel_size, activation=None, q=1, padding='valid', use_bias=True, strides=1, apply_initializer=False):
        super(Oper2DTranspose, self).__init__(name='')

        self.activation = activation
        self.q = q
        self.all_layers = []
        initializer = tf.random_normal_initializer(0., 0.02)
        for i in range(0, q):  # q convolutional layers.
            if not apply_initializer:
                self.all_layers.append(tf.keras.layers.Conv2DTranspose(filters,
                                                                       kernel_size,
                                                                       padding=padding,
                                                                       use_bias=use_bias,
                                                                       strides=strides,
                                                                       activation=activation))
            else:
                self.all_layers.append(tf.keras.layers.Conv2DTranspose(filters,
                                                                       kernel_size,
                                                                       padding=padding,
                                                                       use_bias=use_bias,
                                                                       strides=strides,
                                                                       activation=activation,
                                                                       kernel_initializer=initializer))

    @tf.function
    def call(self, input_tensor, training=False):

        x = self.all_layers[0](input_tensor)  # First convolutional layer.

        if self.q > 1:
            for i in range(1, self.q):
                x += self.all_layers[i](tf.math.pow(input_tensor, i + 1))

        if self.activation is not None:
            return eval('tf.nn.' + self.activation + '(x)')
        else:
            return x


def downsample(filters, size, norm_type='batch_norm', apply_norm=True):
    """Downsamples an input.

    Conv2D => instance_norm => LeakyRelu

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batch_norm' or 'instance_norm'.
      apply_norm: If True, adds the batch_norm layer

    Returns:
      Downsample Sequential Model
    """
    # initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               # kernel_initializer=initializer,
                               use_bias=False))
    if apply_norm:
        if norm_type.lower() == 'batch_norm':
            result.add(tf.keras.layers.batch_normalization())
        elif norm_type.lower() == 'instance_norm':
            result.add(tfa.layers.InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, norm_type='instance_norm', apply_dropout=False):
    """Upsamples an input.

    Conv2DTranspose => instance_norm => Dropout => Relu

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batch_norm' or 'instance_norm'.
      apply_dropout: If True, adds the dropout layer

    Returns:
      Upsample Sequential Model
    """

    # initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        # kernel_initializer=initializer,
                                        use_bias=False))

    if norm_type.lower() == 'batch_norm':
        result.add(tf.keras.layers.batch_normalization())
    elif norm_type.lower() == 'instance_norm':
        result.add(tfa.layers.InstanceNormalization())

    if apply_dropout:
        # result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.Dropout(0.2))

    result.add(tf.keras.layers.ReLU())

    return result

def integrate_oper_upsample(filters, size, norm_type='instance_norm', apply_dropout=False, q=1):
    """Upsamples an input.

    Oper2DTranspose => instance_norm => Dropout => Relu

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batch_norm' or 'instance_norm'.
      apply_dropout: If True, adds the dropout layer

    Returns:
      Upsample Sequential Model
    """
    result = tf.keras.Sequential()
    result.add(
        Oper2DTranspose(filters, size, strides=2, q=q, padding='same', use_bias=False, apply_initializer=False))

    if norm_type.lower() == 'batch_norm':
        result.add(tf.keras.layers.batch_normalization())
    elif norm_type.lower() == 'instance_norm':
        result.add(tfa.layers.InstanceNormalization())

    if apply_dropout:
        # result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.Dropout(0.2))

    # result.add(tf.keras.layers.ReLU())
    # Using tanh instead
    result.add(tf.keras.layers.Activation('tanh'))

    return result


def integrate_oper_downsample(filters, size, norm_type='batch_norm', apply_norm=True, q=1):
    """Downsamples an input.

    Oper2D => instance_norm => LeakyRelu

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batch_norm' or 'instance_norm'.
      apply_norm: If True, adds the batch_norm layer

    Returns:
      Downsample Sequential Model
    """
    result = tf.keras.Sequential()
    result.add(
        Oper2D(filters, size, strides=2, q=q, padding='same', use_bias=False, apply_initializer=False))

    if apply_norm:
        if norm_type.lower() == 'batch_norm':
            result.add(tf.keras.layers.batch_normalization())
        elif norm_type.lower() == 'instance_norm':
            result.add(tfa.layers.InstanceNormalization())

    result.add(tf.keras.layers.Activation('tanh'))

    return result
