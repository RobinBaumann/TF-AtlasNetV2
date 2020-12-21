"""
Created by Robin Baumann <mail@robin-baumann.com> on May 18, 2020.
"""
import tensorflow as tf

from . import encoders, layers

ALLOWED_STRUCTURE_TYPES = ['point', 'patch']
ALLOWED_ADJUSTMENT_TYPES = ['mlp', 'linear']


class AtlasNetV2(tf.keras.Model):

  def __init__(self,
               structure_type="points",
               adjustment_type="mlp",
               structure_dim=3,
               n_structures=10,
               n_latent=1024):
    """
    Instantiates an AtlasNet V2 Model according to  the provided parameters.
    Args:
        structure_type: Type of elementary structures. One of 'point' and 'patch'. Default: 'point'.
        adjustment_type: Type of adjustment module to be used. One of 'mlp' and 'linear'. Default: 'mlp'.
        structure_dim: Dimensionality of the elementary structures. Must be at least 3. Default: 3.
        n_structures: Number of elementary structures to be learned by the model.
        n_latent: number of latent features computed by the encoder network.
    """
    super(AtlasNetV2, self).__init__()

    assert structure_type in ALLOWED_STRUCTURE_TYPES, \
      f"{structure_type} not allowed. Expecting one of {ALLOWED_STRUCTURE_TYPES}."
    assert adjustment_type in ALLOWED_ADJUSTMENT_TYPES, \
      f"{adjustment_type} not allowed. Expecting one of {ALLOWED_ADJUSTMENT_TYPES}."

    self.structure_type = structure_type
    self.adjustment_type = adjustment_type
    self.structure_dim = structure_dim
    self.n_structures = n_structures
    self.n_latent = n_latent

  def build(self, input_shape):
    """See base class for details."""
    self.structure_shape = (
      input_shape[1] // self.n_structures, self.structure_dim)
    self.encoder = encoders.PointNetEncoder(n_latent=self.n_latent)

    if self.structure_type == 'point':
      self.atlas = layers.PointAtlas(units=self.n_structures,
                                     shape=self.structure_shape,
                                     adjustment=self.adjustment_type
                                     )
    else:
      self.atlas = layers.PatchAtlas(units=self.n_structures,
                                     shape=self.structure_shape,
                                     layer_size=128,
                                     adjustment=self.adjustment_type
                                     )

  def call(self, inputs, training=None, mask=None):
    encoding = self.encoder(inputs)
    out_structure, out_patches = self.atlas(encoding)

    return out_structure, out_patches

  def train_step(self, data):
    """
    Overrides train step.
    Args:
        data: batch containing (X,y) pairs.

    Returns: Dictionary containing metrics and loss specified in model.compile()

    """
    x, y = data

    with tf.GradientTape() as tape:
      y_pred, _ = self(x, training=True)
      loss = self.compiled_loss(y, y_pred)

    trainable_variables = self.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    self.compiled_metrics.update_state(y, y_pred)

    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    """
    Overrides evaluation step.
    Args:
        data: Batch of (X,y) pairs.

    Returns: Dictionary containing metrics and loss specified in model.compile()

    """
    x, y = data
    y_pred, _ = self(x, training=False)
    print(y_pred.shape)
    self.compiled_loss(y, y_pred)
    self.compiled_metrics.update_state(y, y_pred)

    return {m.name: m.result() for m in self.metrics}

  def get_config(self):
    return {
      "structure_type": self.structure_type,
      "adjustment_type": self.adjustment_type,
      "structure_dim": self.structure_dim,
      "n_structures": self.n_structures,
      "n_latent": self.n_latent,
      "encoder": self.encoder.get_config(),
      "atlas": self.atlas.get_config(),
    }
