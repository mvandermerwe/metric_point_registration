import trimesh
import numpy as np
import os
import argparse
import pdb
from tqdm import tqdm
import torch
import torch.utils.data as data

import visualize as vis
import utils

import dataset

parser = argparse.ArgumentParser(description='Run ICP on pairs of transformations.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose.')
parser.set_defaults(verbose=False)
args = parser.parse_args()
verbose = args.verbose

# Read config.
cfg = utils.load_config(args.config)
data_folder = cfg['data']['out_dir']
mesh = cfg['data']['meshes'][0]

# Setup dataset.
train_dataset = dataset.RegistrationDataset(
    os.path.join(data_folder, mesh, cfg['data']['pointcloud_file']),
    os.path.join(data_folder, mesh, cfg['data']['train_transforms_file'])
)
train_dataloader = data.DataLoader(
    train_dataset,
    batch_size = cfg['training']['batch_size'],
    shuffle = cfg['training']['shuffle']
)

for batch in train_dataloader:
    vis.visualize_points(batch['points'][0], show=True)
