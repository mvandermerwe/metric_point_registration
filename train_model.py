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
import model

parser = argparse.ArgumentParser(description='Learn a point cloud embedding.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose.')
parser.set_defaults(verbose=False)
args = parser.parse_args()
verbose = args.verbose

# Read config.
cfg = utils.load_config(args.config)
data_folder = cfg['data']['out_dir']
mesh = cfg['data']['meshes'][0]
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

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

# Create model:
registration_model = model.RegistrationNetwork(
    c_dim = cfg['model']['c_dim'],
    dim = cfg['model']['dim'],
    hidden_dim = cfg['model']['hidden_dim'],
    n_points = cfg['data']['num_points'],
    decoder = True,
    device = device
)
print(registration_model)

epoch_it = 0
it = 0

# Training loop
while True:
    epoch_it += 1

    for batch in train_dataloader:
        it += 1

        # TODO: Run training step.
