import trimesh
import numpy as np
import os
import argparse
import pdb
import open3d as o3d
from tqdm import tqdm

import visualize as vis
import utils

parser = argparse.ArgumentParser(description='Run ICP on pairs of transformations.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose.')
parser.set_defaults(verbose=False)
args = parser.parse_args()
verbose = args.verbose

# Read config.
cfg = utils.load_config(args.config)

data_folder = cfg['data']['out_dir']
meshes = cfg['data']['meshes']

for mesh_name in meshes:

    # Load the point cloud we will apply the transforms to.
    points_dict = np.load(os.path.join(data_folder, mesh_name, cfg['data']['pointcloud_file']))
    pointcloud = points_dict['points']

    # Load the test transforms.
    test_transforms = np.load(os.path.join(data_folder, mesh_name, cfg['data']['test_transforms_file']))['transforms']

    for i in range(test_transforms.shape[0]):

        # Turn specific transform into example.
        test_transform = test_transforms[i]
        t1, t2 = utils.array_to_transforms(test_transform)

        # Get the ground truth transform between the given transformations.
        # WARNING: don't trust this, I'm mucking something up.
        t_gt = utils.get_transform_between(t1, t2)

        # Apply each transform to get our two pointclouds.
        pc1 = utils.transform_pointcloud(pointcloud, t1)
        pc2 = utils.transform_pointcloud(pointcloud, t2)

        if verbose:
            vis.visualize_points(pc1, show=True)
            vis.visualize_points(pc2, show=True)
            # vis.visualize_points(utils.transform_pointcloud(pc1, t_gt), show=True)
            # vis.visualize_points_overlay([pc2, utils.transform_pointcloud(pc1, t_gt)], show=True)

        # Converting numpy array point clouds into open3d point cloud objects.
        pc1_o3d = o3d.geometry.PointCloud()
        pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
        pc2_o3d = o3d.geometry.PointCloud()
        pc2_o3d.points = o3d.utility.Vector3dVector(pc2)
        # Visualize using open3d with:
        # o3d.visualization.draw_geometries([pc1_o3d])

        # Maximum correspondence points-pair distance.
        max_corr_dist = 0.02
        # Use ICP to transform pc1 to pc2.
        reg_result = o3d.registration.registration_icp(pc1_o3d, pc2_o3d, max_corr_dist)
        # The transformation matrix is a 4x4 np array.
        icp_trans = reg_result.transformation

        if verbose:
            # Transform p1 with ICP transformation, visualize target p2 and transformed p1.
            trans_pc1 = utils.transform_pointcloud(pc1, icp_trans)
            point_sets = np.array([trans_pc1, pc2])
            vis.visualize_points_overlay(point_sets, show=True)
