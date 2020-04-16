import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vtkplotter import *

def visualize_points(points, bound=0.5, c=None, out_file=None, show=False):
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')

    if len(points) != 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c)

    ax.set_title('Points')
    ax.set_xlim3d(-bound, bound)
    ax.set_ylim3d(-bound, bound)
    ax.set_zlim3d(-bound, bound)
    ax.view_init(elev=30, azim=45)

    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)

def visualize_points_overlay(point_sets, bound=0.5, out_file=None, show=False):
    ''' Visualizes a set of points by overlaying w/ different colors.
    This is especially helpful for ensuring that training data is setup accurately.

    Args:
        point_sets (tensor): list of point cloud data tensors/arrays.
        bound (float): upper/lower bound for the axes.
    '''
    num_sets = len(point_sets)
    colors = ['red', 'blue', 'yellow', 'orange', 'green']
    assert(num_sets <= len(colors))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, s in enumerate(range(num_sets)):
        ax.scatter(point_sets[i][:, 0], point_sets[i][:, 1], point_sets[i][:, 2], c=colors[i])

    ax.set_xlim3d(-bound, bound)
    ax.set_ylim3d(-bound, bound)
    ax.set_zlim3d(-bound, bound)
    ax.view_init(elev=30, azim=45)    

    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)

if __name__ == '__main__':
    mesh = 'data/processed/ffccda82ecc0d0f71740529c616cd4c7/mesh.obj'
    pointcloud = 'data/processed/ffccda82ecc0d0f71740529c616cd4c7/points.npz'
    points = np.load(pointcloud)['points']

    vp = Plotter(N=2, verbose=0)

    plane_mesh = vp.load(mesh, c='grey')
    vp.show(plane_mesh, at=0)

    points = Points(points, c='b')
    vp.show(points, at=1)

    vp.show(interactive=1)
