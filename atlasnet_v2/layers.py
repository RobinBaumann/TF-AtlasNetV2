"""
Created by Robin Baumann <mail@robin-baumann.com> on April 27, 2020.
"""
import tensorflow as tf
from tensorflow.keras import layers

from .adjustment_modules import LinearAdjustment, MLPAdjustment
from .utils import tf_utils

ADJUSTMENTS = ["linear", "mlp"]


class AtlasBase(layers.Layer):
  """
  Base Class for Atlas Layers. Provides adjustment logic.
  """

  def __init__(self, units, shape, deformation_dimensions=3,
               adjustment="linear", **kwargs):
    """
    Base Constructor of Atlas Layers.

    Args:
      units: int, number of elementary structures in layer.
      shape: tuple of (n_points_e, n_dim) where the former specifies the
      number of points per elementary structure and the latter the dimension
      of the elementary structures.
      deformation_dimensions: int, output dimension of adjusted elementary structures,
      defaults to 3.
      adjustment: str, denoting the adjustment module. One of ['linear', 'mlp'].
      Defaults to 'linear'
    """
    super(AtlasBase, self).__init__(**kwargs)
    self.units = units
    self.structure_size = shape
    self.deformation_dimensions = deformation_dimensions
    self.adjustment = adjustment

  def _generate_structures(self, n, shape):
    """
    Generates initial learnable structure points sampled from a 2D unit square.
    Higher dimensions are set to zero initially.

    Args:
        n: Number of structures to be generated.
        shape: tuple containing the shape of the structures as (n_dim, n_struct_points)

    Returns:
        Yields n Tensors with uniformly sampled floats in the first two dimensions.
    """
    n_struct_points, n_dim = shape
    for i in range(0, n):
      random = tf.random.uniform((1, n_struct_points, n_dim), minval=0,
                                 maxval=1)
      mask = tf.concat([tf.ones((1, n_struct_points, 2)),
                        tf.zeros((1, n_struct_points, n_dim - 2))], axis=2)
      structure = random * mask
      assert random.shape == structure.shape
      yield i, structure

  def build(self, input_shape):
    """
    Defer weight creation until input shape is known.

    Args:
        input_shape: out shape of previous layer.
    """
    self.batch_size = input_shape[0]
    if self.adjustment == "linear":
      self.decoder = [
        LinearAdjustment(dimensions=self.deformation_dimensions,
                         n_latent=input_shape[-1]) for _ in range(self.units)
      ]
    else:
      self.decoder = [
        MLPAdjustment(n_latent=input_shape[-1] + self.structure_size[1]) for _
        in range(self.units)
      ]

  def adjust(self, encoding, structure, decoder, adjustment="linear"):
    """
    Performs the adjustment of one elementary structure based on the provided decoder.

    Args:
        encoding: input to the Atlas Layer.
        structure: one elementary structure.
        decoder: decoder for this structure.
        adjustment: string indicating adjustment strategy. One of linear(default) or mlp

    Returns:
        The adjusted structure
    """
    latent = tf.expand_dims(encoding, 1)
    if adjustment == "linear":
      R, t = decoder(latent)
      return tf.add(tf.matmul(structure, R), t)
    else:
      # repeat latent vector of each batch to match number of points in elementary structure
      latent = tf.tile(latent,
                       tf.constant([1, structure.shape[1], structure.shape[2]]))
      # repeat elementary structure to match batch_size
      # Quite hacky, but somehow TF is not in eager mode and thus, this is the simplest way to get the batch size
      shape_tensor = tf.concat(
        [tf.reshape(tf.shape(latent)[0], [-1]), tf.constant([1, 1])], 0)
      exp_struct = tf.tile(structure, shape_tensor)
      # concatenate features of each point with latent vector and decode.
      x = tf.concat([exp_struct, latent], axis=2)
      return decoder(x)

  def get_config(self):
    """Serializes Configuration of AtlasBase Layer."""
    return {
      'units': self.units,
      'structure_size': list(self.structure_size),
      'deformation_dimensions': self.deformation_dimensions,
      'adjustment': self.adjustment,
      'decoder': [d.get_config() for d in self.decoder],
    }


class PointAtlas(AtlasBase):
  """
  Implementation of the Atlas Layer.

  An PointAtlas Layer represents a set of arbitrary structures as point sets.
  Each structure is held within one node of the Layer.
  """

  def __init__(self, units, shape, deformation_dimensions=3,
               adjustment="linear", **kwargs):
    """
    Constructor of PointAtlas Layers.

    Args:
      units: int, number of elementary structures in layer.
      shape: tuple of (n_points_e, n_dim) where the former specifies the
      number of points per elementary structure and the latter the dimension
      of the elementary structures.
      deformation_dimensions: int, output dimension of adjusted elementary structures,
      defaults to 3.
      adjustment: str, denoting the adjustment module. One of ['linear', 'mlp'].
      Defaults to 'linear'
    """
    super(PointAtlas, self).__init__(units, shape, deformation_dimensions,
                                     adjustment)

  def build(self, input_shape):
    """
    Defer weight creation until input shape is known.

    Args:
        input_shape: out shape of previous layer.
    """
    super().build(input_shape)
    self.structures = [
      tf.Variable(s, name=f"ElementaryStructure_{i}") for i, s in
      self._generate_structures(self.units, self.structure_size)
    ]

  def call(self, inputs, **kwargs):
    outputs = []
    out_structures = []
    for i in range(0, self.units):
      out_structures.append(self.structures[i])
      outputs.append(self.adjust(inputs, self.structures[i], self.decoder[i],
                                 self.adjustment))
    return tf.concat(outputs, 1), out_structures

  def get_config(self):
    """Serializes Configuration of PointAtlas Layer."""
    config = super(PointAtlas, self).get_config()
    config.update({'structures': [s.numpy().tolist() for s in self.structures]})
    return config

  @classmethod
  def from_config(cls, config):
    """Deserializes PointAtlas Config."""
    atlas = cls(config['units'], config['shape'],
                config['deformation_dimension'], config['adjustment'])
    atlas.structures = [tf.Variable(s, name=f"ElementaryStructure_{i}") for i, s
                        in enumerate(config['structures'])]
    return atlas


class PatchAtlas(AtlasBase):
  """
  Implementation of the PatchAtlas Layer.

  A PatchAtlas Layer represents a set of arbitrary structures as patches
  """

  def __init__(self, units, shape, layer_size, deformation_dimensions=3,
               tanh=True, adjustment="linear", **kwargs):
    """
    Constructor of PatchAtlas Layers.

    Args:
      units: int, number of elementary structures in layer.
      shape: tuple of (n_points_e, n_dim) where the former specifies the
      number of points per elementary structure and the latter the dimension
      of the elementary structures.
      layer_size: Number of hidden units in PatchDeformation Network.
      deformation_dimensions: int, output dimension of adjusted elementary structures,
      defaults to 3.
      tanh: bool, whether to use tanh Activation in last layer of PatchDeformation Network.
      adjustment: str, denoting the adjustment module. One of ['linear', 'mlp'].
      Defaults to 'linear'
    """
    super(PatchAtlas, self).__init__(units, shape, deformation_dimensions,
                                     adjustment, **kwargs)
    self.layer_size = layer_size
    self.tanh = tanh

  def build(self, input_shape):
    """
    Defer weight creation until input shape is known.

    Args:
        input_shape: out shape of previous layer.
    """
    super().build(input_shape)
    self.deformers = [
      PatchDeformation(self.deformation_dimensions, layer_size=self.layer_size,
                       tanh=self.tanh) for _ in
      range(self.units)
    ]

  def call(self, inputs, **kwargs):
    outputs = []
    patches = []
    for i in range(0, self.units):
      _, patch = next(self._generate_structures(1, self.structure_size))
      patch = self.deformers[i](patch)
      patches.append(patch)

      outputs.append(
        self.adjust(inputs, patch, self.decoder[i], self.adjustment))

    return tf.concat(outputs, 1), patches

  def get_config(self):
    """Serializes Configuration of PatchAtlas Layer."""
    config = super(PatchAtlas, self).get_config()
    config.update({
      'deformers': [d.get_config() for d in self.deformers],
      'layer_size': self.layer_size,
      'tanh': self.tanh
    })
    return config


class PatchDeformation(layers.Layer):
  """
  Patch deformation Layer, which is part of the PatchAtlas Layer.
  """

  def __init__(self, deform_dim=3, layer_size=128, tanh=True, **kwargs):
    """
    Deforms the Patches based on the latent shape signature.
    Args:
      deform_dim: int, number of output dimensions for patches
      layer_size: int, number of hidden units of this layer.
      tanh: bool, whether to use tanh Activation or not.
      **kwargs:
    """
    super(PatchDeformation, self).__init__(**kwargs)

    self.deform_dim = deform_dim
    self.layer_size = layer_size
    self.tanh = tanh

  def build(self, input_shape):
    self.conv1_bn = tf_utils.Conv1DBatchNorm(self.layer_size, 1,
                                             scope="PatchDeform_Conv1D_1")
    self.conv2_bn = tf_utils.Conv1DBatchNorm(self.layer_size, 1,
                                             scope="PatchDeform_Conv1D_2")
    self.conv3 = layers.Conv1D(self.deform_dim, 1,
                               activation="tanh" if self.tanh else None)

  def call(self, inputs, **kwargs):
    x = self.conv1_bn(inputs)
    x = self.conv2_bn(x)
    x = self.conv3(x)

    return x

  def get_config(self):
    """Serializes Configuration of PatchDeformation Layer."""
    return {
      'deform_dim': self.deform_dim,
      'layer_size': self.layer_size,
      'tanh': self.tanh
    }
