"""
Created by Robin Baumann <mail@robin-baumann.com> on May 30, 2020.
"""

import tensorflow as tf


def __chamfer_distances(y_true, y_pred):
  # B x N x M x D Tensor, where entry n,m is a D-dimensional vector of element-wise differences
  # (y_true_n - y_pred_m)
  difference = (
    tf.expand_dims(y_true, axis=-2) -
    tf.expand_dims(y_pred, axis=-3))
  # Square distances between the point sets. Results in : B x M x N Tensor of squared differences.
  # | y_true_n - y_pred_m |^2
  squared_diff = tf.einsum("...i,...i->...", difference, difference)

  diff_true2pred = tf.reduce_min(squared_diff, axis=-1)
  diff_pred2true = tf.reduce_min(squared_diff, axis=-2)

  return diff_true2pred, diff_pred2true


def chamfer_loss(y_true, y_pred, sample_weight=None):
  """
  Computes symmetrical Chamfer Distance between predicted output shapes and ground truth shape.

  Args:
      y_true: float32 Tensor of shape [B, N, D] where B is a batch dimension.
      y_pred: float32 Tensor of shape [B, M, D] of predicted output shape (concatenated elementary structures).

  Returns:
      float32 Tensor of shape [B] containing the symmetric chamfer distances per batch.
  """
  diff_true2pred, diff_pred2true = __chamfer_distances(y_true, y_pred)
  return tf.reduce_sum(diff_true2pred, axis=-1) + tf.reduce_sum(diff_pred2true,
                                                                axis=-1)


def mean_chamfer_loss(y_true, y_pred, sample_weight=None):
  """
  Mean-normalized variant of chamfer distane.

  Args:
      y_true: float32 Tensor of shape [B, N, D] where B is a batch dimension.
      y_pred: float32 Tensor of shape [B, M, D] of predicted output shape (concatenated elementary structures).
      sample_weight:

  Returns:
      float32 Tensor of shape [B] containing the symmetric chamfer distances per batch.
  """
  diff_true2pred, diff_pred2true = __chamfer_distances(y_true, y_pred)
  return tf.reduce_mean(diff_true2pred, axis=-1) + tf.reduce_mean(
    diff_pred2true, axis=-1)
