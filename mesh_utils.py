import trimesh
import numpy as np
import pdb
import os
import sys
import math

import visualize as vis

def check_mesh_contains(mesh, points):
    return mesh.contains(points)

def center_mesh(mesh):
    center_transform = np.eye(4)
    center_transform[0,3] = -mesh.centroid[0]
    center_transform[1,3] = -mesh.centroid[1]
    center_transform[2,3] = -mesh.centroid[2]

    mesh.apply_transform(center_transform)
    return mesh

def load_mesh(mesh_filename, center=True):
    '''
    Helper that loads and centers mesh.
    '''
    mesh = trimesh.load(mesh_filename)

    # print("Wt: ", mesh.is_watertight)
    if center:
        return center_mesh(mesh)
    else:
        return mesh
    
def scale_mesh(mesh):
    '''
    Scale the given mesh to fit in a 1x1x1 box.
    '''
    # Find the maximum dimension size to determine how we need to scale the object.
    lo, hi = mesh.bounds
    max_dim = max(hi[0], abs(lo[0]), hi[1], abs(lo[1]), hi[2], abs(lo[2]))

    # Scale object to be in 1x1x1 box. To do this, scale to be inside slightly smaller 1/1.03 bounding box (same as DeepSDF).
    scale = ((1/1.03)*0.5) / max_dim
    mesh.apply_scale(scale)
    
    return mesh, scale

def sample_occupancy(mesh, uniform_sample_count, surface_sample_count, bound=0.5, verbose=False):
    '''
    Assumes already scaled and within the given bounds (i.e., [-bound, bound] on each axis).

    Samples both randomly and on the surface of the object.
    '''
    
    points = np.random.rand(uniform_sample_count, 3) - bound
    
    surface_points, surface_normals = sample_surface(mesh, surface_sample_count//3)

    # Move points along normals slightly in both directions.
    surface_up = np.add(surface_points, 0.0025*surface_normals)
    surface_down = np.add(surface_points, -0.0025*surface_normals)

    points = np.concatenate((points, surface_up, surface_down, surface_points))
    occ = check_mesh_contains(mesh, points)

    if verbose:
        occ_pts = points[occ]
        n_pts = len(occ_pts)
        vis_idx = np.random.choice(n_pts, 1024)        
        vis.visualize_points(occ_pts[vis_idx], bound=bound, show=True)

    return points, occ

def sample_sdf(mesh, uniform_sample_count, surface_sample_count, bound=0.5, verbose=False):
    '''
    Assumes already scaled and within the given bounds (i.e., [-bound, bound] on each axis).
    '''
    
    points = np.random.rand(uniform_sample_count, 3) - bound
    
    surface_points, surface_normals = sample_surface(mesh, surface_sample_count//3)

    # Move points along normals slightly in both directions.
    surface_up = np.add(surface_points, 0.0025*surface_normals)
    surface_down = np.add(surface_points, -0.0025*surface_normals)

    points = np.concatenate((points, surface_up, surface_down, surface_points))
    sdf = -trimesh.proximity.signed_distance(mesh, points)

    if verbose:
        n_pts = len(points)
        vis_idx = np.random.choice(n_pts, 1024)
        vis.visualize_points(points[vis_idx], bound=bound, c=sdf[vis_idx], show=True)

    return points, sdf

def sample_surface(mesh, sample_count, bound=0.5, noise=0.0, verbose=False):
    '''
    Sample from surface. Bound is for viz.
    '''

    # Sample the surface pointcloud.
    pointcloud, pointcloud_idx = mesh.sample(sample_count, return_index=True)
    pointcloud_normals = mesh.face_normals[pointcloud_idx]

    # Generate and add noise to pointcloud.
    a = np.random.rand(sample_count, 3) * noise
    pointcloud = pointcloud + a

    if verbose:
        vis.visualize_points(pointcloud[:1000], bound=bound, show=True)

    return pointcloud, pointcloud_normals
