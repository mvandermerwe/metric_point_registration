import trimesh
import numpy as np
import os
import argparse
import pdb
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

import utils

class RegistrationDataset(Dataset):

    def __init__(self, pointcloud_file, transforms_file, transform=None):
        '''
        Create a new registration dataset.

        Args:
            pointcloud_file (str): file with the base pointcloud.
            transforms_file (str): file with all transforms in it.
            transform (callable, optional): optional transform on a sample.
        '''

        self.pointcloud = np.load(pointcloud_file)['points'].astype(np.float32)
        self.transformations = np.load(transforms_file)['transforms'].astype(np.float32)
        self.transform = transform

    def __len__(self):
        return self.transformations.shape[0]

    def __getitem__(self, idx):
        data = {}

        t_idx = utils.array_to_transform(self.transformations[idx])

        data['tranform'] = t_idx
        data['points'] = utils.transform_pointcloud(self.pointcloud, t_idx)

        if self.transform is not None:
            data = self.transform(data)

        return data
