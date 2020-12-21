"""
Created by Robin Baumann <mail@robin-baumann.com> on May 05, 2020.
"""

import tensorflow as tf
from atlasnet_v2 import encoders


class PointNetEncoderTest(tf.test.TestCase):
    def test_pointnet_shape(self):
        data = tf.random.uniform((2, 10, 3))  # 2 batches, 10 3D points each.
        pointnet = encoders.PointNetEncoder(n_latent=64)
        result = pointnet(data)
        print(result.shape)
        assert result.shape == (2, 64)