import trimesh
import numpy as np
import os
import sys
import argparse
import pdb
from tqdm import tqdm, trange
import torch
import torch.optim as optim

import visualize as vis
import utils
import lie_group as lie

import model

parser = argparse.ArgumentParser(description='Run ICP on pairs of transformations.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose.')
parser.set_defaults(verbose=False)
args = parser.parse_args()
verbose = args.verbose

# Read config.
cfg = utils.load_config(args.config)

data_folder = cfg['data']['out_dir']
meshes = cfg['data']['meshes']
out_dir = cfg['training']['out_dir']
vis_dir = cfg['align']['vis_dir']
vis_every = cfg['align']['vis_every']
max_iters = cfg['align']['max_iters']
epsilon = cfg['align']['epsilon']
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

# Load our model:
registration_model = model.RegistrationNetwork(
    c_dim = cfg['model']['c_dim'],
    dim = cfg['model']['dim'],
    hidden_dim = cfg['model']['hidden_dim'],
    n_points = cfg['data']['num_points'],
    decoder = True,
    device = device
)
registration_model = registration_model.to(device)
registration_model.eval()
# Load model + optimizer if exists.
model_dict = {
    'model': registration_model,
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
else:
    print('Could not find checkpoint.')
    sys.exit()

for mesh_name in meshes:

    # Load the point cloud we will apply the transforms to.
    points_dict = np.load(os.path.join(data_folder, mesh_name, cfg['data']['pointcloud_file']))
    pointcloud = points_dict['points']
    
    # Load the test transforms.
    test_transforms = np.load(os.path.join(data_folder, mesh_name, cfg['data']['test_transforms_file']))['transforms']

    for i in trange(test_transforms.shape[0]):
        # Make vis directory.
        vis_dir_i = os.path.join(vis_dir, 'align_%03d' % i)
        if not os.path.exists(vis_dir_i):
            os.makedirs(vis_dir_i)

        # Turn specific transform into example.
        test_transform = test_transforms[i]
        t1, t2 = utils.array_to_transforms(test_transform)

        # Get the ground truth transform between the given transformations.
        # WARNING: don't trust this, I'm mucking something up.
        t_gt = utils.get_transform_between(t1, t2)

        # Apply each transform to get our two pointclouds.
        pc1 = utils.transform_pointcloud(pointcloud, t1)
        pc2 = utils.transform_pointcloud(pointcloud, t2)

        pc1_cuda = torch.from_numpy(pc1).to(device).float().unsqueeze(0)
        pc2_cuda = torch.from_numpy(pc2).to(device).float().unsqueeze(0)

        # Get the embedding of pointcloud 2 to try to match.
        with torch.no_grad():
            pc2_result_dict = registration_model(pc2_cuda)
            c_goal = pc2_result_dict['c']

        # Rodrigues vector is a succint and differentiable representation of 3D rigid body transforms.
        rodrigues = torch.zeros(3, device=device, requires_grad=True)
        translation = torch.zeros(3, device=device, requires_grad=True)

        # Setup the optimizer to solve the optimization.
        optimizer = optim.Adam([rodrigues, translation], lr=0.01)

        if verbose:
            vis.visualize_points(pc1, show=True)
            vis.visualize_points(pc2, show=True)
        
        for i in range(max_iters):
            optimizer.zero_grad()

            rotation_matrix = lie.SO3_exp(rodrigues)

            transform_pc1 = torch.transpose(torch.matmul(rotation_matrix, torch.transpose(pc1_cuda[0], 0, 1)), 0, 1) + translation

            result_dict = registration_model(transform_pc1.unsqueeze(0))
            c_transform = result_dict['c']

            loss_c = torch.nn.MSELoss()(c_transform, c_goal)
            loss_c.backward()

            optimizer.step()

            if (i % vis_every) == 0:
                vis.visualize_points_overlay([pc2, result_dict['p'].squeeze(0).detach().cpu()],
                                             out_file=os.path.join(vis_dir_i, 'align_%03d' % i))

            if (loss_c < epsilon):
                break
