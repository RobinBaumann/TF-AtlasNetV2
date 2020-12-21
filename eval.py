"""
Created by Robin Baumann <mail@robin-baumann.com> on June 23, 2020.
"""
import os

from absl import app
from absl import flags
from absl import logging
import json

import tensorflow as tf
from atlasnet_v2 import loss, model
from data.shape_net import provider
from data.humans import hdataset_provider

# Hyperparameters
flags.DEFINE_integer("n_points", 2500,
                     "Number of points to sample from each structure.")
flags.DEFINE_integer("batch_size", 16, "Size of the mini batches.")

# Data and Environment configuration
flags.DEFINE_integer("gpu", -1, "Device index of GPU. Default: -1 = No GPU.")
flags.DEFINE_string("data", None, "Path to ShapeNet dataset.")
flags.DEFINE_list("categories", None,
                  "List of categories used in training. If None, all categories are used.")
flags.DEFINE_string("model", None,
                    "Path to the model that should be evaluated.")
flags.DEFINE_string("dataset", "shapenet",
                    "Dataset to train on. One of [shapenet, hdataset].")
flags.DEFINE_string("result_file", "results.json", "Filepath, to which evaluation results should be saved.")

# AtlasNet Config
flags.DEFINE_string("structure_type", "point",
                    "Representation of elementary structures. One of [`points`, `patch`]")
flags.DEFINE_string("adjustment", "mlp",
                    "Adjustment Module. One of [`mlp`, `linear`].")

FLAGS = flags.FLAGS

# Required: Data directory:
flags.mark_flag_as_required("data")
flags.mark_flag_as_required("model")


def evaluate_model(_):
  if FLAGS.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
  else:
    logging.info(
      "Not using a GPU. Specify one by passing --gpu {idx} with idx >= 0.")

  if FLAGS.dataset == "hdataset":
    data_index = hdataset_provider.HDatasetGenerator.initialize_hdataset(
      FLAGS.data)
    eval_index = data_index[~data_index['train']]

    eval_generator = hdataset_provider.HDatasetGenerator(eval_index)
  else:
    data_index = provider.ShapeNetGenerator.initialize_shapenet(FLAGS.data,
                                                                class_choice=FLAGS.categories)
    eval_index = data_index[~data_index['train']]

    eval_generator = provider.ShapeNetGenerator(eval_index, n_points=FLAGS.n_points)

  eval_dataset = eval_generator.create_dataset().batch(FLAGS.batch_size)

  atlas_net = model.AtlasNetV2(structure_type=FLAGS.structure_type,
                               adjustment_type=FLAGS.adjustment)

  logging.info(
    f"Using {len(eval_generator)} samples for evaluation.")

  atlas_net.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=loss.chamfer_loss,
                    metrics=['mean_squared_error', loss.mean_chamfer_loss])

  atlas_net.predict(eval_dataset, steps=1)
  atlas_net.load_weights(FLAGS.model)

  results = atlas_net.evaluate(eval_dataset,
                     steps=len(eval_generator) // FLAGS.batch_size, return_dict=True
                     )

  results['model_metadata'] = {
    'num_params': atlas_net.count_params(),
    'structure_type': FLAGS.structure_type,
    'adjustment_type': FLAGS.adjustment,
  }

  results_json = json.dumps(results)

  with tf.io.gfile.GFile(FLAGS.result_file, "w") as outfile:
    json.dump(results_json, outfile)


if __name__ == "__main__":
  app.run(evaluate_model)
