import yaml
import pdb
import numpy as np
import os
import matplotlib.pyplot as plt
import transforms3d
import kaolin
import torch

def load_config(config_file):
    with open(config_file, 'r') as f:
        cfg = yaml.load(f)
    return cfg

def transform_pointcloud(points, t):
    '''
    Transform the given pointcloud by t (4x4 matrix).
    '''
    points_temp = np.ones([points.shape[0], 4])
    points_temp[:,:3] = points

    return np.dot(t, points_temp.T).T[:,:3]

def array_to_transforms(array):
    '''
    Convert array to two transforms. Each six should be
    a transformation.
    '''
    assert(array.shape[0]==12)
    
    return array_to_transform(array[:6]), array_to_transform(array[6:])

def array_to_transform(array):
    '''
    Convert array to 4x4 transform matrix.
    Array should be [ai, aj, ak, x, y, z] where ai,aj,ak are euler angles for static xyz
    and x,y,z is translation vector.
    '''
    assert(array.shape[0]==6)

    rot = transforms3d.euler.euler2mat(array[0], array[1], array[2])

    transform = np.eye(4)
    transform[:3,:3] = rot
    transform[0,3] = array[3]
    transform[1,3] = array[4]
    transform[2,3] = array[5]

    return transform

def inverse_transform(t):
    '''
    Get the inverse of the provided transformation.
    '''
    rot = t[:3,:3]
    trans = np.array([t[0,3], t[1,3], t[2,3]])
    trans_inv = np.dot(-rot.T, trans.T).T

    t_inv = np.eye(4)
    t_inv[:3,:3] = rot.T
    t_inv[0,3] = trans_inv[0]
    t_inv[1,3] = trans_inv[1]
    t_inv[2,3] = trans_inv[2]
    return t_inv

def get_transform_between(t1, t2):
    '''
    Get the ground truth transformation between t1 and t2.
    t1 and t2 should be w.r.t a common frame, which will
    be used for the resulting transform.
    '''
    return np.matmul(t2, inverse_transform(t1))

def chamfer_distance(pc_A, pc_B, device=None):
    pc_A = torch.from_numpy(pc_A).to(device).float()
    pc_B = torch.from_numpy(pc_B).to(device).float()
    dist = kaolin.metrics.point.chamfer_distance(pc_A, pc_B)
    return dist
