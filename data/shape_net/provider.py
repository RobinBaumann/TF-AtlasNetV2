"""
Created by Robin Baumann <mail@robin-baumann.com> on April 13, 2020.
"""
import os

import numpy as np
import pandas as pd
import open3d as o3d
import tensorflow as tf


class ShapeNetGenerator:
    """
    Lazily load point clouds and annotations from filesystem and prepare it for model training.
    """

    def __init__(self, dataset, n_points=2500, visualize=False):
        """
        Instantiate a data provider instance for ShapeNet dataset.
        Args:
            dataset: pandas DataFrame containing the index to the files (train or test set)
            batch_size: the desired batch size
            n_points: the amount of points to sample per instance.
            visualize: Return Open3D PointCloud data structure instead of numpy array for better visualization.
        """
        self.dataset = dataset
        self.n_points = n_points
        self.visualize = visualize

        self.indices = np.arange(0, len(dataset), dtype=np.uint32)

        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Called at the end of each epoch. Performs shuffling of the index to the dataset.
        """
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

    def __flow(self):
        """
        Generates one sample at a time indefinitely. Shuffles index after every sample was drawn once.

        Yields:
            One sampled point cloud as numpy array of shape (, n_points, 3).
        """

        while True:
            for i, pcd_file in self.dataset.iterrows():
                pcd = o3d.io.read_point_cloud(pcd_file['pointcloud'])

                if self.visualize:
                    sample = pcd.uniform_down_sample(int(len(pcd.points) / self.n_points))
                else:
                    points = np.asarray(pcd.points, dtype=np.float32)
                    sample = points[np.random.choice(points.shape[0], size=self.n_points, replace=False)]

                yield sample, sample

            self.on_epoch_end()

    def create_dataset(self):
        return tf.data.Dataset.from_generator(
            self.__flow,
            (
                tf.float32, tf.float32
            ),
            (
                tf.TensorShape([self.n_points, 3]),
                tf.TensorShape([self.n_points, 3])
            )
        )

    def __len__(self):
        """
        Computes the number of steps required to load each file at least once.

        Return:
            number of steps required to load each file at least once
        """
        return len(self.dataset)

    @staticmethod
    def initialize_shapenet(data_dir, class_choice=None, train_frac=0.8):
        """
        Loads an index to all files and structures them.

        Args:
            data_dir: Directory containing the ShapeNet download from download.sh
            class_choice: List of strings containing names of classes. If None (default), all classes will be loaded.
            train_frac: Scalar in [0,1) denoting the size of the training data.

        Returns:
            pandas data frame containing an index to all files and a label index,
            mapping numerical label representations to label names.
        """

        rendered_root = os.path.join(data_dir, "ShapeNet/ShapeNetRendering/")
        pointcloud_root = os.path.join(data_dir, "customShapeNet/")
        catfile = os.path.join(data_dir, "synsetoffset2category.txt")
        categories = {}
        print(os.getcwd())
        print(os.listdir("."))
        with open(catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                categories[ls[0]] = ls[1]
        if class_choice is not None:
            categories = {k: v for k, v in categories.items() if k in class_choice}

        print("Categories: \n \t {}".format(categories))

        df = pd.DataFrame()

        for key, category in categories.items():
            dir_rendered = os.path.join(rendered_root, category)
            rendered = sorted(os.listdir(dir_rendered))
            dir_pointcloud = os.path.join(pointcloud_root, category, 'ply')
            pointclouds = [os.path.join(dir_pointcloud, f) for f in sorted(os.listdir(dir_pointcloud))
                           if f.endswith('.points.ply') and not f.startswith("*")
                           and f.split('.')[0] in rendered]

            rendered = [os.path.join(dir_rendered, path) for path in rendered
                        if os.path.join(dir_pointcloud, path + '.points.ply') in pointclouds]
            pc_per_cat = len(rendered)
            print("[Category {}] \t {} total files \t {}%".format(
                key,
                pc_per_cat,
                pc_per_cat/float(len(dir_rendered))
            ))

            train_mask = np.zeros(shape=pc_per_cat).astype(np.bool)
            train_mask[:int(train_frac*pc_per_cat)] = True

            assert(len(pointclouds) == len(train_mask))
            df = df.append(pd.DataFrame({
                'id': rendered,
                'pointcloud': pointclouds,
                'category': [key] * pc_per_cat,
                'train': train_mask,
            }), ignore_index=True)

        return df
