import numpy as np
import pyvista as pv
import matplotlib
import matplotlib.pyplot as plt
import os
from os.path import join as osj
from os.path import exists as ose
import pyacvd
import open3d as o3d  
import shutil
import mcubes
import sys



def get_volume_pos(pts1,ct, mode = 'matrix'):
    """
    input:  pts1: np array of pixel values on the ct grid wrt the ct pixel indices 
            ct: vti file for the ct image 
    output: boundary_pts: a sequence of eucleandian coordinates of the shell points 
    """
    inds_1 = np.argwhere(pts1 == 1.0)# ; print(inds_1.shape) # get indeices of "1" pixels 

    if mode == 'forloop':
        # for loop mode 
        volume_pts = []
        for ele in inds_1:
            # print(ele)
            target = [0.0,0.0,0.0]
            ct.TransformIndexToPhysicalPoint(ele,target)
            # ct.TransformIndexToPhysicalPoint(ele[np.array([2,1,0])],target)
            volume_pts.append(np.array(target))
            # volume_pts.append(np.array(target)[np.array([0,1,2])])
        volume_pts = np.asarray(volume_pts)
    elif mode == 'matrix':
        # matrix mode 
        volume_pts = (inds_1*np.array(ct.spacing))+np.array(ct.origin)
    else: 
        sys.exit('error, no such mode')
    return volume_pts


def get_shell_pos(pts1,ct):
    """
    input:  pts1: np array of pixel values on the ct grid wrt the ct pixel indices 
            ct: vti file for the ct image 
    output: boundary_pts: a sequence of eucleandian coordinates of the shell points 
    """
    inds_1 = np.argwhere(pts1 == 1.0)# ; print(inds_1.shape) # get indeices of "1" pixels 
    pts_pad = np.zeros((pts1.shape[0]+2,pts1.shape[1]+2,pts1.shape[2]+2))
    pts_pad[1:-1, 1:-1, 1:-1] = pts1
    def get_neighbor_value(i,j,k, pts = pts_pad):
        i+=1;j+=1;k+=1
        # get neighbour vlaues of a given pixel, calculate product of them 
        return pts_pad[i-1,j,k]*pts_pad[i+1,j,k]*pts_pad[i,j-1,k]*pts_pad[i,j+1,k]*pts_pad[i,j,k-1]*pts_pad[i,j,k+1]


    shell_pts = []
    for ele in inds_1: 
        # print(ele)
        if get_neighbor_value(ele[0],ele[1],ele[2])==0.0:
            target = [0.0,0.0,0.0]
            ct.TransformIndexToPhysicalPoint(ele,target)
            shell_pts.append(target)
    shell_pts = np.asarray(shell_pts)
    return shell_pts


def laplacian(pts1, kernel_size= 3, threshold = 0.5):    
    pad_dim = int(kernel_size-1/2)
    pts_pad = np.zeros((pts1.shape[0]+pad_dim*2,pts1.shape[1]+pad_dim*2,pts1.shape[2]+pad_dim*2))
    pts_pad[pad_dim:-pad_dim, pad_dim:-pad_dim, pad_dim:-pad_dim] = pts1
    import torch
    pts_pad_ts = torch.tensor(pts_pad,dtype = torch.float32).unsqueeze(0).unsqueeze(0)
    Flap = torch.nn.Conv3d(1,1, kernel_size, stride=1, padding=0, 
                    dilation=1, bias=False, 
                    padding_mode='zeros'
    )
    kernel = torch.ones((kernel_size,kernel_size,kernel_size))*(1/kernel_size**3)
    with torch.no_grad():
        Flap.weight.copy_(kernel)
    # print(Flap.weight.data)
    pts_smooth = Flap(pts_pad_ts).detach().numpy()
    return np.squeeze(1*((pts_smooth>=threshold)))


def get_isosurf(npy, iso_value = 0.5):
    # X, Y, Z = np.mgrid[:npy.shape[0], :npy.shape[1], :npy.shape[2]]
    vertices, triangles = mcubes.marching_cubes(npy, iso_value)
    faces = (np.hstack(((np.ones(len(triangles))*3)[...,None],triangles))).astype(int)
    import pyvista as pv 
    mesh = pv.PolyData(vertices, faces = np.ravel(faces))
    return mesh



def ind2phy(id_set,ct):
    # this is slow, becuase of the for loop 
    pts = id_set.copy()
    for i,ele in enumerate(id_set):
        # print(ele)
        target = [0.0,0.0,0.0]
        ct.TransformContinuousIndexToPhysicalPoint(ele,target)
        # ct.TransformIndexToPhysicalPoint(ele[np.array([2,1,0])],target)
        pts[i] = target
    return pts


def ind2phy_fast(id_set,ct):
    # this is slow, becuase of the for loop 
    #TODO add directions , currently defualt direction 100 010 001
    return np.asarray(ct.origin)+ id_set*np.asarray(ct.spacing)

def phy2ind_fast(id_set,ct):
    # this is slow, becuase of the for loop 
    #TODO add directions , currently defualt direction 100 010 001
    return (id_set - np.asarray(ct.origin))* (1/np.asarray(ct.spacing))


def meshind2phy(mesh,ct):
    # this is slow, becuase of the for loop 
    pts = mesh.points
    pts_new = pts.copy()
    for i,ele in enumerate(pts):
        # print(ele)
        target = [0.0,0.0,0.0]
        ct.TransformContinuousIndexToPhysicalPoint(ele,target)
        # ct.TransformIndexToPhysicalPoint(ele[np.array([2,1,0])],target)
        pts_new[i] = target
    mesh.points = pts_new
        # volume_pts.append(np.array(target)[np.array([0,1,2])])
    return mesh




def ptc2surf_poisson_write(ptc, normal_knn = 10, write_mesh= False, ofile = None, temp_dir = './o3d_temp',remesh_params = [3,10000], **poisson_config):
    """
    input:  ptc: point cloud <N,3>
            **poisson_config: configuration of the o3d poisson reconstruction
    output: remesh: reconstructed surface 
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptc)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd], point_show_normal=False)
    pcd.orient_normals_consistent_tangent_plane(normal_knn ) # seems like 10 is a sweet spot for normals
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    
    # poisson ######################################################################
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, **poisson_config)[0] 
    print('poisson reconstruction done!')
    # o3d.visualization.draw_geometries([pcd,poisson_mesh])

    # if ose(temp_dir):
    #     shutil.rmtree(temp_dir)
    # os.mkdir(temp_dir)
    o3d.io.write_triangle_mesh(osj(temp_dir,'temp_shell.ply'),poisson_mesh)
    poisson_mesh_pv = pv.read(osj(temp_dir,'temp_shell.ply'))
    poisson_mesh_pv.save(osj(temp_dir,'temp_shell.vtp'))

    clus = pyacvd.Clustering(poisson_mesh_pv)
    # # mesh is not dense enough for uniform remeshing
    clus.subdivide(remesh_params[0]);clus.cluster(remesh_params[1])
    print('remeshing done!')
    remesh = clus.create_mesh()# remesh
    # remesh.save(osj(temp_dir,'temp_shell_remesh.vtp'))
    if write_mesh == True: 
        remesh.save(ofile)
    return remesh 
    #############################################################################

def ptc2surf_poisson(ptc, temp_dir = './', normal_knn = 10, **poisson_config):
    """
    input:  ptc: point cloud <N,3>
            **poisson_config: configuration of the o3d poisson reconstruction
    output: remesh: reconstructed surface 
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptc)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd], point_show_normal=False)
    pcd.orient_normals_consistent_tangent_plane(normal_knn) # seems like 10 is a sweet spot for normals
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    
    # poisson ######################################################################
    # print(pcd)
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, **poisson_config)[0] 
    print('poisson reconstruction done!')
    # o3d.visualization.draw_geometries([pcd,poisson_mesh])
    # help(poisson_mesh)
    pts = np.asarray(poisson_mesh.vertices)
    tris = np.asarray(poisson_mesh.triangles)
    # o3d.io.write_triangle_mesh(osj(temp_dir,'temp_shell.ply'),poisson_mesh)
    # poisson_mesh_pv = pv.read(osj(temp_dir,'temp_shell.ply'))
    faces = np.hstack((np.ones(len(tris))[...,None]*3, tris))
    poisson_mesh_pv = pv.PolyData(pts, faces = np.ravel(faces.astype(int)))
    # os.remove(osj(temp_dir,'temp_shell.ply'))
    return poisson_mesh_pv


    # clus = pyacvd.Clustering(poisson_mesh_pv)
    # # # mesh is not dense enough for uniform remeshing
    # clus.subdivide(remesh_params[0]);clus.cluster(remesh_params[1])
    # print('remeshing done!')
    # remesh = clus.create_mesh()# remesh

    # return remesh 



def ACVD(input_mesh, keep_pts=None,  keep_ratio=None, subdivide = 3):
    N = len(input_mesh.points)

    if keep_pts ==None: 
        keep_pts = int(keep_ratio * N)

    clus = pyacvd.Clustering(input_mesh)
    clus.subdivide(subdivide);clus.cluster(keep_pts)
    remesh = clus.create_mesh()
    return remesh