"""
Created by Robin Baumann <mail@robin-baumann.com> on May 13, 2020.
"""
import tensorflow as tf
from atlasnet_v2 import model, loss
from data.shape_net import provider

BATCH_SIZE = 2
N_POINTS = 5000
STRUCTS = 10
POINTS_PER_STRUCT = N_POINTS // STRUCTS


class AtlasNetV2Test(tf.test.TestCase):

    def test_point_atlas_model_creation(self):
        atlasnet = model.AtlasNetV2(structure_type='points', adjustment_type='linear', n_structures=10)
        rand_input = tf.random.uniform((BATCH_SIZE, N_POINTS, 3))

        out_shape, structs = atlasnet(rand_input)

        assert structs[0].shape == (1, POINTS_PER_STRUCT, 3)
        assert out_shape.shape == (BATCH_SIZE, N_POINTS, 3)

    def test_patch_atlas_model_creation(self):
        atlasnet = model.AtlasNetV2(structure_type='patches', adjustment_type='linear', n_structures=10)

        rand_input = tf.random.uniform((BATCH_SIZE, N_POINTS, 3))

        out_shape, structs = atlasnet(rand_input)

        assert structs[0].shape == (1, POINTS_PER_STRUCT, 3)
        assert out_shape.shape == (BATCH_SIZE, N_POINTS, 3)

    def test_fit_with_real_input_and_serialization(self):
        df = provider.ShapeNetGenerator.initialize_shapenet("../data/shape_net/", class_choice=['plane'], train_frac=0.7)

        atlasnet = model.AtlasNetV2(structure_type='point', adjustment_type='linear', n_structures=10)
        generator = provider.ShapeNetGenerator(df, n_points=10000, visualize=False)
        dataset = generator.create_dataset().batch(BATCH_SIZE)
        atlasnet.compile(optimizer='Adam', loss=loss.chamfer_loss, metrics=['mean_squared_error'])
        print("Model compiled successfully!")
        atlasnet.fit(dataset, epochs=1, steps_per_epoch=1)

        atlasnet.save_weights('weights/atlasnet')
        import json
        print(json.dumps(atlasnet.get_config()))

    def test_load_saved_model(self):
        atlasnet = model.AtlasNetV2(structure_type='patch', adjustment_type='linear', n_structures=10)
        atlasnet.compile(optimizer='Adam', loss=loss.chamfer_loss, metrics=['mean_squared_error'])
        df = provider.ShapeNetGenerator.initialize_shapenet("../data/shape_net/", class_choice=['plane'], train_frac=0.7)
        generator = provider.ShapeNetGenerator(df, n_points=10000, visualize=False)
        dataset = generator.create_dataset().batch(BATCH_SIZE)
        atlasnet.evaluate(dataset, steps=1)
        atlasnet.load_weights('weights/atlasnet')
        atlasnet.evaluate(dataset, steps=1)
