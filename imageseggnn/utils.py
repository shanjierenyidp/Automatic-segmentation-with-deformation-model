
import torch 
import numpy as np
import pyvista as pv
from imageseggnn.mesh_operations.interpolation import *

def get_e(pts,ct_grad_ts,xyz2ind):
    '''
    pts: input points <N,3>
    '''
    inds = xyz2ind.normalize((pts))#;
    lb = torch.tensor([0.5,0.5,0.5],device = inds.device)
    ub = torch.tensor(ct_grad_ts.shape[1:-1],device = inds.device) -1.5
    inds = torch.clamp(inds,lb,ub)
    values = interpolate(ct_grad_ts, inds.unsqueeze(0))
    return torch.sum(values)

def get_ptse(pts,ct_grad_ts,xyz2ind):
    inds = xyz2ind.normalize((pts))#; print(inds.shape)
    values = interpolate(ct_grad_ts, inds.unsqueeze(0))
    return values[0,:,-1]

def create_vtp(my_mesh,xyz2hat, ct_grad_ts,xyz2ind): 
    new_ptse = get_ptse(xyz2hat.denormalize(my_mesh.verts_packed()), ct_grad_ts,xyz2ind)
    temp_faces = np.concatenate((np.ones(len(my_mesh.faces_packed()))[...,None]*3 , my_mesh.faces_packed().detach().cpu().numpy()), axis = -1).ravel()
    temp_points = xyz2hat.denormalize(my_mesh.verts_packed()).detach().cpu().numpy()
    temp_mesh = pv.PolyData(temp_points, faces= temp_faces.astype(int) )
    #only mag
    temp_mesh.point_data['e'] = new_ptse.detach().cpu().numpy()
    return temp_mesh

def load_vtp(fn):
    print('read in vtp')
    temp = pv.read(fn)
    pts = torch.tensor(temp.points, dtype = torch.float32)
    faces = torch.tensor(temp.faces.reshape(-1,4)[:,1:], dtype = torch.int64)
    print(pts.shape, faces.shape)
    return pts, faces 

def get_std(pts, ct_ts, xyz2ind): 
    inds = xyz2ind.normalize((pts))#; print(inds.shape)
    values = interpolate(ct_ts, inds.unsqueeze(0))
    return torch.std(values[0])