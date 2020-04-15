import trimesh
import numpy as np
import os
import argparse
import pdb
from tqdm import tqdm, trange
import torch
import torch.utils.data as data
import torch.optim as optim
import kaolin
from tensorboardX import SummaryWriter

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

# Shorthands:
out_dir = cfg['training']['out_dir']
lr = cfg['training']['learning_rate']
visualize_every = cfg['training']['visualize_every']
validate_every = cfg['training']['validate_every']
backup_every = cfg['training']['backup_every']
print_every = cfg['training']['print_every']
vis_dir = os.path.join(out_dir, 'vis')

# Output + vis directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Setup datasets.
train_dataset = dataset.RegistrationDataset(
    os.path.join(data_folder, mesh, cfg['data']['pointcloud_file']),
    os.path.join(data_folder, mesh, cfg['data']['train_transforms_file'])
)
train_dataloader = data.DataLoader(
    train_dataset,
    batch_size = cfg['training']['batch_size'],
    shuffle = cfg['training']['shuffle']
)
validation_dataset = dataset.RegistrationDataset(
    os.path.join(data_folder, mesh, cfg['data']['pointcloud_file']),
    os.path.join(data_folder, mesh, cfg['data']['validation_transforms_file'])
)
val_dataloader = data.DataLoader(validation_dataset, batch_size=10, shuffle=True)
vis_dataloader = data.DataLoader(validation_dataset, batch_size=10, shuffle=True)
data_vis = next(iter(vis_dataloader))

# Create model:
registration_model = model.RegistrationNetwork(
    c_dim = cfg['model']['c_dim'],
    dim = cfg['model']['dim'],
    hidden_dim = cfg['model']['hidden_dim'],
    n_points = cfg['data']['num_points'],
    decoder = True,
    device = device
)
registration_model = registration_model.to(device)
print(registration_model)

# Get optimizer
optimizer = optim.Adam(registration_model.parameters(), lr=lr)

# Load model + optimizer if exists.
model_dict = {
    'model': registration_model,
    'optimizer': optimizer
}
model_file = os.path.join(out_dir, 'model.pt')
if os.path.exists(model_file):
    print('Loading checkpoint from local file...')
    state_dict = torch.load(model_file, map_location='cpu')

    for k, v in model_dict.items():
        if k in state_dict:
            v.load_state_dict(state_dict[k])
        else:
            print('Warning: Could not find %s in checkpoint' % k)

    load_dict = {k: v for k, v in state_dict.items()
                 if k not in model_dict}
else:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get('val_loss_best', np.inf)

# Training loop
while True:
    epoch_it += 1

    for batch in train_dataloader:
        it += 1
        
        registration_model.train()
        optimizer.zero_grad()

        # Get data and move to device.
        pointcloud = batch.get('points').to(device).float()

        # Forward pass.
        result_dict = registration_model(pointcloud)

        # Loss.
        loss = 0.0
        for i in range(cfg['training']['batch_size']):
            loss += kaolin.metrics.point.chamfer_distance(pointcloud[i], result_dict['p'][i])
        loss.backward()
        optimizer.step()

        logger.add_scalar('loss', loss, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))

        # Visualize.
        if visualize_every > 0 and (it % visualize_every) == 0:
            print('Visualizing.')
            registration_model.eval()
            p_src = data_vis.get('points').to(device).float()
            with torch.no_grad():
                p_out = registration_model(p_src)['p']
            p_src = p_src.cpu()
            p_out = p_out.cpu()
                
            for i in trange(10):
                vis.visualize_points(p_src[i], out_file=os.path.join(vis_dir, '%03d_in.png' % i))
                vis.visualize_points(p_out[i], out_file=os.path.join(vis_dir, '%03d_out.png' % i))

        # Validate.
        if validate_every > 0 and (it % validate_every) == 0:
            print('Validating.')
            registration_model.eval()
            val_loss = 0.0
            val_it = 0.0
            for val_batch in tqdm(val_dataloader):
                p_in = val_batch.get('points').to(device).float()
                with torch.no_grad():
                    p_out = registration_model(pointcloud)['p']
                for i in range(10):
                    loss += kaolin.metrics.point.chamfer_distance(p_in[i], p_out[i])
                val_loss += loss / 10.0
                val_it += 1
            val_loss = val_loss / float(val_it)
            logger.add_scalar('val_loss', val_loss, it)

            if val_loss < metric_val_best:
                metric_val_best = val_loss
                print('Saving new best model. Loss=%03f' % metric_val_best)
                torch.save({
                    'model': registration_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch_it': epoch_it,
                    'it': it,
                    'val_loss_best': metric_val_best
                }, os.path.join(out_dir, 'model_best.pt'))

        # Backup.
        if backup_every > 0 and (it % backup_every) == 0:
            torch.save({
                'model': registration_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch_it': epoch_it,
                'it': it,
                'val_loss_best': metric_val_best
            }, os.path.join(out_dir, 'model.pt'))
