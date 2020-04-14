import trimesh
import numpy as np
import os
import argparse
import pdb
from tqdm import tqdm

import visualize as vis
import mesh_utils
import utils

parser = argparse.ArgumentParser(description='Create point clouds from ShapeNet objs.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose.')
parser.set_defaults(verbose=False)
args = parser.parse_args()

# Read config.
cfg = utils.load_config(args.config)

shapenet_dir = cfg['data']['shapenet_dir']
out_dir = cfg['data']['out_dir']
meshes = cfg['data']['meshes']
num_points = cfg['data']['num_points']
bound = cfg['data']['bound']
num_train_transforms = cfg['data']['num_train_transforms']
num_test_transforms = cfg['data']['num_test_transforms']
rotation_max = cfg['data']['rotation_max']
translation_max = cfg['data']['translation_max']

# Setup out.
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# For each mesh make a folder to store the points.
for mesh_name in tqdm(meshes):
    mesh_out_dir = os.path.join(out_dir, str(mesh_name).replace('/', '_'))

    if not os.path.exists(mesh_out_dir):
        os.mkdir(mesh_out_dir)

    # Load centered mesh and scale.
    try:
        mesh = mesh_utils.load_mesh(os.path.join(shapenet_dir, mesh_name + '.off'))
    except:
        print("Couldn't find model: " + str(mesh_name))
        continue
    mesh, s = mesh_utils.scale_mesh(mesh)

    # Sample a pointcloud from the surface.
    pointcloud, normals = mesh_utils.sample_surface(mesh, num_points, bound=bound, verbose=args.verbose)

    # Save.
    np.savez(os.path.join(mesh_out_dir, cfg['data']['pointcloud_file']), points=pointcloud, normals=normals, scale=s)

    # Generate train transformations:
    transforms = np.random.random([num_train_transforms, 6])

    # Scale rotational and translational components.
    transforms[:,:3] = transforms[:,:3] * rotation_max
    transforms[:,3:] = transforms[:,3:] * translation_max

    np.savez(os.path.join(mesh_out_dir, cfg['data']['train_transforms_file']), transforms=transforms)

    # Generate test transformations:
    transforms = np.random.random([num_test_transforms, 12])

    # Scale rotational and translational components.
    transforms[:,:3] = transforms[:,:3] * rotation_max
    transforms[:,3:6] = transforms[:,3:6] * translation_max
    transforms[:,6:9] = transforms[:,6:9] * rotation_max
    transforms[:,9:] = transforms[:,9:] * translation_max

    np.savez(os.path.join(mesh_out_dir, cfg['data']['test_transforms_file']), transforms=transforms)
