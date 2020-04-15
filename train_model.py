import trimesh
import numpy as np
import os
import argparse
import pdb
from tqdm import tqdm

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

for i in range(len(train_dataset)):
    sample = train_dataset[i]
    vis.visualize_points(sample['points'], show=True)
