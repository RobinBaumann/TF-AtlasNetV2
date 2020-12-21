"""
Created by Robin Baumann <mail@robin-baumann.com> on June 06, 2020.
"""
import os
from datetime import datetime

import mlflow
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from atlasnet_v2 import loss, model
from data.shape_net import provider

# Hyperparameters
flags.DEFINE_float("val_split", 0.2, "Fraction of data which should be used for evaluation in each epoch.")
flags.DEFINE_integer("n_points", 2500, "Number of points to sample from each structure.")
flags.DEFINE_integer("epochs", 100, "Number of epochs to train.")
flags.DEFINE_integer("batch_size", 16, "Size of the mini batches.")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate of Optimizer.")

# Data and Environment configuration
flags.DEFINE_integer("gpu", -1, "Device index of GPU. Default: -1 = No GPU.")
flags.DEFINE_string("data", None, "Path to ShapeNet dataset.")
flags.DEFINE_string("checkpoint", "./ckpt/weights.{epoch:03d}-{val_loss:.2f}.h5",
                    "Path to save the model after every epoch.")
flags.DEFINE_string("output", "./AtlasNet_v2_weights.h5", "Model output file.")
flags.DEFINE_string("logdir", "./logs", "Directory for TensorBoard Logs.")
flags.DEFINE_string("experiment", "Experiment", "Name of Experiment which gets logged to MLflow.")
flags.DEFINE_list("categories", None, "List of categories used in training. If None, all categories are used.")
flags.DEFINE_string("fromcheckpoint", None, "Optionally load weights from file and continue training.")

# AtlasNet Config
flags.DEFINE_string("structure_type", "point", "Representation of elementary structures. One of [`points`, `patch`]")
flags.DEFINE_string("adjustment", "mlp", "Adjustment Module. One of [`mlp`, `linear`].")

FLAGS = flags.FLAGS

# Required: Data directory:
flags.mark_flag_as_required("data")

# Validators:
flags.register_validator("val_split",
                         lambda value: 0 <= value <= 1,
                         message="val_split must be in range (0,1)",
                         flag_values=FLAGS)


def random_split(samples, frac):
  logging.info(f"Reserving {frac * len(samples)} files for validation.")
  samples = samples.sample(frac=1).reset_index(drop=True)
  pivot = max(1, int(frac * len(samples)))
  return samples.iloc[:pivot], samples.iloc[pivot:]


def train_model(_):
  if FLAGS.fromcheckpoint is None:
    now = datetime.now()
    mlflow.set_experiment(f"{FLAGS.experiment} vom {now.strftime('%m/%d/%Y, %H:%M:%S')}")
  else:
    mlflow.set_experiment(FLAGS.experiment)

  with mlflow.start_run():
    if FLAGS.gpu >= 0:
      os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
      os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    else:
      logging.info("Not using a GPU. Specify one by passing --gpu {idx} with idx >= 0.")

    data_index = provider.ShapeNetGenerator.initialize_shapenet(FLAGS.data,
                                                                class_choice=FLAGS.categories)
    train_index = data_index[data_index['train']]
    train_files, val_files = random_split(train_index, 1-FLAGS.val_split)
    train_generator = provider.ShapeNetGenerator(train_files, n_points=FLAGS.n_points)
    eval_generator = provider.ShapeNetGenerator(val_files, n_points=FLAGS.n_points)

    train_dataset = train_generator.create_dataset().batch(FLAGS.batch_size)
    eval_dataset = eval_generator.create_dataset().batch(FLAGS.batch_size)

    logging.info(f"Using {len(train_generator)} samples for training and {len(eval_generator)} samples for evaluation.")

    atlas_net = model.AtlasNetV2(structure_type=FLAGS.structure_type, adjustment_type=FLAGS.adjustment)

    callbacks = [
      tf.keras.callbacks.TensorBoard(FLAGS.logdir, write_graph=False),
      tf.keras.callbacks.ModelCheckpoint(FLAGS.checkpoint, save_weights_only=True, save_best_only=True),
      MLflowCallback(),
    ]

    atlas_net.compile(optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate),
                      loss=loss.chamfer_loss,
                      metrics=['mean_squared_error', loss.mean_chamfer_loss])

    if FLAGS.fromcheckpoint is not None:
      atlas_net.fit(train_dataset, steps_per_epoch=1, epochs=1)
      atlas_net.load_weights(FLAGS.from_checkpoint)

    atlas_net.fit(train_dataset,
                  steps_per_epoch=len(train_generator) // FLAGS.batch_size,
                  epochs=FLAGS.epochs,
                  callbacks=callbacks,
                  validation_data=eval_dataset,
                  validation_steps=len(eval_generator) // FLAGS.batch_size
                  )

    mlflow.log_param('AtlasNet-Config', atlas_net.get_config())

    atlas_net.save_weights(FLAGS.output)


class MLflowCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    mlflow.log_metrics({"train_loss": logs['loss'],
                        "val_loss": logs['val_loss'],
                        "val_mean_chamfer_distance": logs['val_mean_chamfer_loss'],
                        "val_mean_squared_error": logs["val_mean_squared_error"]})


if __name__ == "__main__":
  app.run(train_model)
