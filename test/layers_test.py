"""
Created by Robin Baumann <mail@robin-baumann.com> on May 13, 2020.
"""

import tensorflow as tf
from atlasnet_v2 import layers

BATCH_SIZE = 2
LATENT_SIZE = 64
N_POINTS = 5000
STRUCTS = 5
POINTS_PER_STRUCT = N_POINTS // STRUCTS


class PointAtlasTest(tf.test.TestCase):

    def test_linear_adjust_structure_shape(self):
        latent = tf.random.uniform((BATCH_SIZE, LATENT_SIZE)) # 2 batches, 64 features each
        point_atlas_lin_adj = layers.PointAtlas(STRUCTS, shape=(POINTS_PER_STRUCT, 3), adjustment="linear")

        out_shape, out_structs = point_atlas_lin_adj(latent)

        assert len(point_atlas_lin_adj.structures) == STRUCTS
        assert point_atlas_lin_adj.structures[0].shape == (1, POINTS_PER_STRUCT, 3)
        assert len(out_structs) == STRUCTS
        assert out_structs[0].shape == point_atlas_lin_adj.structures[0].shape
        assert out_shape.shape == (BATCH_SIZE, N_POINTS, 3)

    def test_mlp_adjust_structure_shape(self):
        latent = tf.random.uniform((BATCH_SIZE, LATENT_SIZE)) # 2 batches, 64 features each
        point_atlas_mlp_adj = layers.PointAtlas(STRUCTS, shape=(POINTS_PER_STRUCT, 3), adjustment="mlp")

        out_shape, out_structs = point_atlas_mlp_adj(latent)
        assert len(point_atlas_mlp_adj.structures) == STRUCTS
        assert len(out_structs) == STRUCTS
        assert out_structs[0].shape == (1, POINTS_PER_STRUCT, 3)
        assert out_shape.shape == (BATCH_SIZE, N_POINTS, 3)


class PatchAtlasTest(tf.test.TestCase):

    def test_linear_adjust_structure_shape(self):
        latent = tf.random.uniform((BATCH_SIZE, LATENT_SIZE))# 2 batches, 64 features each
        patch_atlas_lin_adj = layers.PatchAtlas(STRUCTS, shape=(POINTS_PER_STRUCT, 3), layer_size=128, adjustment="linear")

        out_shape, out_structs = patch_atlas_lin_adj(latent)
        assert len(out_structs) == STRUCTS
        assert out_structs[0].shape == (1, POINTS_PER_STRUCT, 3)
        assert out_shape.shape == (BATCH_SIZE, N_POINTS, 3)

    def test_mlp_adjust_structure_shape(self):
        latent = tf.random.uniform((BATCH_SIZE, LATENT_SIZE)) # 2 batches, 64 features each
        patch_atlas_mlp_adj = layers.PatchAtlas(STRUCTS, shape=(POINTS_PER_STRUCT, 3), layer_size=128, adjustment="mlp")

        out_shape , out_structs = patch_atlas_mlp_adj(latent)
        assert len(out_structs) == STRUCTS
        assert out_structs[0].shape == (1, POINTS_PER_STRUCT, 3)
        assert out_shape.shape == (BATCH_SIZE, N_POINTS, 3)
