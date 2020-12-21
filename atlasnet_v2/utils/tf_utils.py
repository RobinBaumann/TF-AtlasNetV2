"""
Created by Robin Baumann <mail@robin-baumann.com> on April 27, 2020.
"""

import tensorflow as tf
from tensorflow.keras import layers


class DenseBatchNorm(layers.Layer):
  """
  Utility class to apply Dense + Batch Normalization.
  """

  def __init__(self, units, use_bias=True, scope=None, activation='relu'):
    super(DenseBatchNorm, self).__init__()

    self.scope = scope
    self.units = units
    self.use_bias = use_bias
    self.act_fun = activation

  def build(self, input_shape):
    self.dense = layers.Dense(self.units, use_bias=self.use_bias,
                              input_shape=input_shape)
    self.bn = layers.BatchNormalization(momentum=0.9)
    self.activation = layers.Activation(self.act_fun)

  def call(self, inputs, **kwargs):
    with tf.name_scope(self.scope):
      x = self.dense(inputs)
      x = self.bn(x)
      x = self.activation(x)
    return x


class Conv1DBatchNorm(layers.Layer):
  """
  Utility class to apply Convolution + Batch Normalization.
  """

  def __init__(self, filters, kernel_size, padding='same', strides=1,
               use_bias=True, scope=None, activation='relu'):
    super(Conv1DBatchNorm, self).__init__()

    self.scope = scope
    self.filters = filters
    self.kernel_size = kernel_size
    self.padding = padding
    self.strides = strides
    self.use_bias = use_bias
    self.act_fun = activation

  def build(self, input_shape):
    self.conv1d = layers.Conv1D(filters=self.filters,
                                kernel_size=self.kernel_size,
                                strides=self.strides,
                                padding=self.padding,
                                use_bias=self.use_bias,
                                input_shape=input_shape
                                )
    self.bn = layers.BatchNormalization(momentum=0.9)
    self.activation = layers.Activation(self.act_fun)

  def call(self, inputs, **kwargs):
    with tf.name_scope(self.scope):
      x = self.conv1d(inputs)
      x = self.bn(x)
      x = self.activation(x)
    return x
