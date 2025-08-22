import pyacvd
import numpy as np 
from scipy.signal import convolve


def remesh(mesh, N, subdivide = 3): 
    clus = pyacvd.Clustering(mesh)
    # mesh is not dense enough for uniform remeshing
    clus.subdivide(subdivide)
    clus.cluster(N)
    # remesh
    remesh = clus.create_mesh()
    return remesh 


import vtk 
from torch_geometric.nn import knn
import torch
import pyvista as pv
def cellArrayDivider(input_array):
    # divied vtk cell array into individual cells 
    N = len(input_array)
    cursor = 0
    head_id = []
    segs = []
    
    while(cursor<N):
        head_id.append(cursor)
        segs.append(input_array[cursor+1:cursor+input_array[cursor]+1])
        cursor = cursor+input_array[cursor]+1
        # print(cursor)
    return segs

def extract_landmarks(mesh,landmark = [6,1,3,2,4,5]):
    edges = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, feature_angle=30.0)
    stripper = vtk.vtkStripper()
    stripper.SetInputData(edges)
    stripper.Update()
    # Convert back to PyVista PolyData
    stripped_edges = pv.wrap(stripper.GetOutput())
    stripped_edges_list = cellArrayDivider(stripped_edges.lines)

    mesh.point_data['landmark']  = 0
    for i in range(len(stripped_edges_list)):
        xx = torch.tensor(stripped_edges.points[stripped_edges_list[i]])
        yy = torch.tensor(mesh.points)
        assign_index = knn(yy, xx, 1)
        mesh.point_data['landmark'][assign_index[1]] = '{:d}'.format(landmark[i])
    return mesh 

def make_tube(centerline_points, tube_radius, N = 100):
    centerline = pv.Spline(centerline_points, N)
    tube_mesh = centerline.tube(radius=tube_radius)
    return tube_mesh


def gradient_sobel(image):
    image = image.detach().cpu().numpy()
    # kernel_x = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    #                      [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    #                      [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])
    # kernel_y = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    #                      [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    #                      [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])
    # kernel_z = np.array([[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
    #                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #                      [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])

    kernel_x = np.array([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]])
    
    kernel_y = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])
    
    kernel_z = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                            [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                            [[1, 0, -1], [2, 0, -2], [1, 0, -1]]])

    grad_x = convolve(image, kernel_x, mode='same')
    grad_y = convolve(image, kernel_y, mode='same')
    grad_z = convolve(image, kernel_z, mode='same')

    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    grad_dir = np.arctan2(grad_y, grad_x)

    grad_x, grad_y, grad_z, grad_mag, grad_dir = torch.from_numpy(grad_x), torch.from_numpy(
        grad_y), torch.from_numpy(grad_z), torch.from_numpy(grad_mag), torch.from_numpy(grad_dir)

    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min())
    return grad_x, grad_y, grad_z, grad_mag, grad_dir





