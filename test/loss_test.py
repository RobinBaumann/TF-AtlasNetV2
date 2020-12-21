"""
Created by Robin Baumann <mail@robin-baumann.com> on May 30, 2020.
"""
import tensorflow as tf
from atlasnet_v2 import loss
import numpy as np

BATCH_SIZE = 2
N_POINTS = 5000


class ChamferLossTest(tf.test.TestCase):

    def test_loss(self):
        dim = 3
        unit_cube = tf.constant(2*((np.arange(2**dim)[:, None] & (1 << np.arange(dim))) > 0) - 1)
        tw_cube = 2 * unit_cube
        expected_result = 48.
        out = loss.chamfer_loss(unit_cube, tw_cube)
        out_rev = loss.chamfer_loss(tw_cube, unit_cube)

        self.assertAllClose(expected_result, out, rtol=1e-6)
        self.assertAllClose(expected_result, out_rev, rtol=1e-6)