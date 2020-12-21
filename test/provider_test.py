"""
Created by Robin Baumann <mail@robin-baumann.com> on April 20, 2020.
"""

import tensorflow as tf
from data.shape_net import provider


class ProviderTest(tf.test.TestCase):

    def test_shapenet_provider_len(self):
        dataset = provider.ShapeNetGenerator.initialize_shapenet("../data/shape_net/", class_choice=['plane'])
        generator = provider.ShapeNetGenerator(dataset, n_points=42)

        assert len(dataset) == len(generator)

    def test_shapenet_provider_batch_size(self):
        dataset = provider.ShapeNetGenerator.initialize_shapenet("../data/shape_net/", class_choice=None)
        generator = provider.ShapeNetGenerator(dataset, n_points=42)
        data = iter(generator.create_dataset().batch(8))
        assert next(data)[0].shape == (8, 42, 3)
