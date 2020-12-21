"""
Created by Robin Baumann <mail@robin-baumann.com> on May 03, 20.
"""
import tensorflow as tf
from tensorflow.keras import layers

from .utils import tf_utils


class LinearAdjustment(layers.Layer):
  """
  Implementation of the linear adjustment module.
  """

  def __init__(self, dimensions=3, n_latent=1024):
    """
    Constructor of LinearAdjustment Module.
    Args:
        dimensions: int, number of output dimensions.
        n_latent: int, length of latent shape signature from Encoder.
    """
    super(LinearAdjustment, self).__init__()

    self.n_latent = n_latent
    self.dim = dimensions

  def build(self, input_shape):
    self.conv1_bn = tf_utils.Conv1DBatchNorm(filters=self.n_latent // 2,
                                             kernel_size=1,
                                             scope="LinearAdjust_Conv1D_1")
    self.conv2_bn = tf_utils.Conv1DBatchNorm(filters=self.n_latent // 2,
                                             kernel_size=1,
                                             scope="LinearAdjust_Conv1D_2")
    self.conv3 = layers.Conv1D(filters=(self.dim + 1) * 3, kernel_size=1,
                               activation="tanh")

  def call(self, inputs, **kwargs):
    x = self.conv1_bn(inputs)
    x = self.conv2_bn(x)
    x = self.conv3(x)
    R = tf.reshape(tf.slice(x, [0, 0, 0], [-1, -1, self.dim * 3]),
                   (tf.shape(x)[0], self.dim, 3))
    t = tf.reshape(tf.slice(x, [0, 0, self.dim * 3], [-1, -1, -1]),
                   (tf.shape(x)[0], 1, 3))

    return R, t

  def get_config(self):
    return {
      'n_latent': self.n_latent,
      'dim': self.dim
    }


class MLPAdjustment(layers.Layer):
  """
  Implementation of the MLP adjustment module.
  """

  def __init__(self, n_latent=1024):
    """
    Constructor of MLPAdjustment Layer.
    Args:
        n_latent: int, length of latent shape signature from Encoder.
    """
    super(MLPAdjustment, self).__init__()
    self.n_latent = n_latent

    self.conv1_bn = tf_utils.Conv1DBatchNorm(n_latent, 1,
                                             scope="MLPAdjust_Conv1D_1")
    self.conv2_bn = tf_utils.Conv1DBatchNorm(n_latent // 2, 1,
                                             scope="MLPAdjust_Conv1D_2")
    self.conv3_bn = tf_utils.Conv1DBatchNorm(n_latent // 4, 1,
                                             scope="MLPAdjust_Conv1D_3")
    self.conv4 = layers.Conv1D(3, 1, activation="tanh")

  def call(self, inputs, **kwargs):
    x = self.conv1_bn(inputs)
    x = self.conv2_bn(x)
    x = self.conv3_bn(x)
    x = self.conv4(x)

    return x

  def get_config(self):
    return {
      'n_latent': self.n_latent
    }
