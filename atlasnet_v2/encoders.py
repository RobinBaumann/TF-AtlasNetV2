"""
Created by Robin Baumann <mail@robin-baumann.com> on April 27, 2020.
"""
from tensorflow.keras import layers

from .utils import tf_utils


class PointNetEncoder(layers.Layer):
  """
  Encoder Network based on PointNet.
  """

  def __init__(self, n_latent=1024):
    """
    Constructor.

    Args:
        n_latent: Number of features in the latent space (after the max pooling)
    """
    super(PointNetEncoder, self).__init__()

    self.n_latent = n_latent

  def build(self, input_shape):
    self.bn1 = tf_utils.Conv1DBatchNorm(filters=64, kernel_size=1,
                                        scope="Encoder_BatchNormConv1D_1")
    self.bn2 = tf_utils.Conv1DBatchNorm(filters=128, kernel_size=1,
                                        scope="Encoder_BatchNormConv1D_2")
    self.bn3 = tf_utils.Conv1DBatchNorm(filters=self.n_latent, kernel_size=1,
                                        activation=None,
                                        scope="Encoder_BatchNormConv1D_3")
    self.maxpool = layers.GlobalMaxPooling1D(data_format='channels_last')
    self.bn4 = tf_utils.DenseBatchNorm(units=self.n_latent,
                                       scope="Encoder_BatchNormDense")

  def call(self, inputs, **kwargs):
    """
    Implementation of forward pass.

    Args:
        inputs: Layer input Tensor

    Returns:
         latent vector of shape (batch_size, n_point, n_latent)
    """
    x = self.bn1(inputs)
    x = self.bn2(x)
    x = self.bn3(x)
    x = self.maxpool(x)
    x = self.bn4(x)

    return x

  def get_config(self):
    return {'n_latent': self.n_latent}
