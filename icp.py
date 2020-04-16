import trimesh
import numpy as np
import os
import argparse
import pdb
import open3d as o3d
import torch
from tqdm import tqdm, trange
import time

import visualize as vis
import utils

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
vis_dir = cfg['align']['vis_dir']
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

for mesh_name in meshes:

    # Load the point cloud we will apply the transforms to.
    points_dict = np.load(os.path.join(data_folder, mesh_name, cfg['data']['pointcloud_file']))
    pointcloud = points_dict['points']

    # Load the test transforms.
    test_transforms = np.load(os.path.join(data_folder, mesh_name, cfg['data']['test_transforms_file']))['transforms']

    # Record information on optimization runs.
    times = []
    chamfer_distances = []
    gt_cloud_distances = []
    rmse_distances = []
    
    # for i in tqdm([16, 20, 21, 26, 27, 36, 38, 39, 41, 49, 50, 52, 54, 56, 59, 62, 67,
    #                72, 73, 77, 78, 85, 87, 89, 94, 95]):
    for i in trange(test_transforms.shape[0]):

        # Make vis directory.
        vis_dir_i = os.path.join(vis_dir, 'align_%03d' % i)
        if not os.path.exists(vis_dir_i):
            os.makedirs(vis_dir_i)

        # Turn specific transform into example.
        test_transform = test_transforms[i]
        t1, t2 = utils.array_to_transforms(test_transform)

        # Get the ground truth transform between the given transformations.
        t_gt = utils.get_transform_between(t1, t2)

        # Apply each transform to get our two pointclouds.
        pc1 = utils.transform_pointcloud(pointcloud, t1)
        pc2 = utils.transform_pointcloud(pointcloud, t2)

        # if verbose:
            # vis.visualize_points(pc1, show=True)
            # vis.visualize_points(pc2, show=True)
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
        max_corr_dist = 0.3

        start = time.time()
        
        # Use ICP to transform pc1 to pc2.
        reg_result = o3d.registration.registration_icp(pc1_o3d, pc2_o3d, max_corr_dist,
                                                       criteria=o3d.registration.ICPConvergenceCriteria(
                                                           max_iteration=1000
                                                       ))
        end = time.time()
        opt_time = end - start
        
        # The transformation matrix is a 4x4 np array.
        icp_trans = reg_result.transformation

        # Transform p1 with ICP transformation, visualize target p2 and transformed p1.
        trans_pc1 = utils.transform_pointcloud(pc1, icp_trans)

        c_d = utils.chamfer_distance(pc2, trans_pc1, device=device)

        gt_cloud_distance = torch.nn.MSELoss()(torch.from_numpy(pc2), torch.from_numpy(trans_pc1))

        times.append(opt_time)
        chamfer_distances.append(c_d.item())
        gt_cloud_distances.append(gt_cloud_distance)
        rmse_distances.append(reg_result.inlier_rmse)

        if verbose:
            point_sets = np.array([trans_pc1, pc2])
            vis.visualize_points_overlay(point_sets, out_file=os.path.join(vis_dir_i, 'result_icp.png'))

    np.savez(os.path.join(vis_dir, 'results_icp.npz'),
             times=np.array(times),
             chamfer_distances=np.array(chamfer_distances),
             gt_cloud_distances=np.array(gt_cloud_distances),
             rmse_distances=np.array(rmse_distances))
