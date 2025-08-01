import os
from os.path import join as osj
import re
import numpy as np
import pyvista as pv
from tqdm import tqdm
import sys
sys.path.insert(1, '../')
from imageseggnn.mesh_operations.normalizer import *
from imageseggnn.mesh_operations.transforms import *
from imageseggnn.mesh_operations.interpolation import *
from imageseggnn.utils import get_e

import torch 
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.io import load_ply, save_ply
from pytorch3d.structures import Meshes
import pylab
import matplotlib.pyplot as plt

params = {'legend.fontsize': 25,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
pylab.rcParams.update(params)

device = torch.device("cuda:0")
## read in filename list of the data
postprocess_path = "../Data/CT_data"

file_name_list = os.listdir(osj(postprocess_path,'npys'))
file_name_list = sorted(file_name_list, key=lambda x:float(re.findall("(\d+)",x)[0]))
N = len(file_name_list); print('total num of data:',N)
# print(file_name_list)

original_path = "../Data/original"
result_path = "./results/experiement1_deform"

sind = 0 # 0,1 
sids = [63,68]
sidstrs = ['0063_1001','0068_0001']

output_path = osj(result_path,'GNN_LDDMM_id00{:d}_exp1_ref'.format(sids[sind])) # play
src_filename = [osj(result_path,'GNN_LDDMM_id00{:d}_exp1'.format(sids[sind])),'mesh_3000.ply'] # this should be pre-smoothed mesh 

src_filename_copy = osj(output_path,src_filename[-1])
if ose(output_path):
    shutil.rmtree(output_path)
    
os.mkdir(output_path)
shutil.copy(osj(*src_filename), src_filename_copy)


i = sind  # change index name

# read in the ct data 
ct_name = os.listdir(osj(original_path,
                file_name_list[i][:-4],
                'Images'))
ct = pv.read(osj(original_path,
                    file_name_list[i][:-4],
                    'Images',ct_name[0]))

# read in the grads 
ct_ts = torch.load(osj(postprocess_path,'ct_gradients',file_name_list[i][:-4]+'.pt'))[0,0,...].unsqueeze(0).unsqueeze(-1)
ct_grad_ts_pack = torch.load(osj(postprocess_path,'ct_gradients',file_name_list[i][:-4]+'_grads_grads2.pt'))
ct_grad_ts = ct_grad_ts_pack[0,3].unsqueeze(0).unsqueeze(-1)
ct_grad2_ts = ct_grad_ts_pack[1,3].unsqueeze(0).unsqueeze(-1)

ct_ts = ct_ts.to(device)#; print(ct_ts.shape)
ct_grad_ts = ct_grad_ts.to(device)#; print(ct_grad_ts.shape)
ct_grad2_ts = ct_grad2_ts.to(device)#; print(ct_grad2_ts.shape)
    
# create transition functions using normalizer 
xyz2ind = Normalizer_ts(params = [torch.tensor(ct.origin, device = device), torch.tensor(ct.spacing, device = device)],method = 'ms', dim=0)

# load in the label surface 
verts, faces = load_ply(src_filename_copy)
faces = faces.to(device)
verts = verts.to(device)

# create normalizer for the mesh 
xyz2hat = Normalizer_ts(method = 'ms', dim=0)
verts_hat = xyz2hat.fit_normalize(verts)
src_mesh = Meshes(verts=[verts_hat], faces=[faces])

e0 = get_e(xyz2hat.denormalize(src_mesh.verts_packed()), ct_grad_ts,xyz2ind) # initial energy 

# also calulate the energy of the label 
# read in the label mesh 
label_mesh_vtp = pv.read(osj(original_path,
                    file_name_list[i][:-4],
                    'Meshes',file_name_list[i][:-4]+'.vtp'))
label_verts = torch.tensor(label_mesh_vtp.points, dtype = torch.float32, device = device)
label_faces = torch.tensor(label_mesh_vtp.faces.reshape(-1,4)[:,1:], dtype = torch.float32, device = device)


###########################################################################################################
# create normalizer for the mesh 
label_xyz2hat = Normalizer_ts(method = 'ms', dim=0)
label_verts_hat = xyz2hat.normalize(label_verts)
label_mesh = Meshes(verts=[label_verts_hat], faces=[label_faces])
label_dpos = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
label_optimizer = torch.optim.Adam([label_dpos], lr=1e-3,weight_decay=1e-4 )



chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []
# optimize
for j in tqdm(range(3001)):
    # Initialize optimizer
    label_optimizer.zero_grad()
    
    # Deform the mesh
    new_src_mesh = src_mesh.offset_verts(label_dpos)
    
    # We sample 5k points from the surface of each mesh 
    sample_trg = sample_points_from_meshes(label_mesh, 10000)
    sample_src = sample_points_from_meshes(new_src_mesh, 10000)

    # We compare the two sets of pointclouds by computing (a) the chamfer loss
    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(new_src_mesh)
    
    # mesh normal consistency
    loss_normal = mesh_normal_consistency(new_src_mesh)
    
    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

        # Weighted sum of the losses
    loss = loss_chamfer * 10.0 + loss_edge * 1.0 + loss_normal * 0.01 + loss_laplacian * 0.1

    # Save the losses for plotting
    chamfer_losses.append(float(loss_chamfer.detach().cpu()))
    edge_losses.append(float(loss_edge.detach().cpu()))
    normal_losses.append(float(loss_normal.detach().cpu()))
    laplacian_losses.append(float(loss_laplacian.detach().cpu()))

    # Optimization step
    loss.backward()
    label_optimizer.step()

    if j % 500 ==0:

        # save_ply(osj(output_path,'label_mesh_{:d}.ply'.format(j)), 
        #             label_xyz2hat.denormalize(new_src_mesh.verts_packed()), new_src_mesh.faces_packed())
        save_ply(osj(output_path,'label_mesh_{:d}.ply'.format(j)), 
                    xyz2hat.denormalize(new_src_mesh.verts_packed()), new_src_mesh.faces_packed())
        
        print('plotting')

        fig, ax = plt.subplots(figsize=(10,8))
        lw = 4
        er1 = torch.tensor(chamfer_losses)
        er2 = torch.tensor(edge_losses)
        er3 = torch.tensor(normal_losses)
        er4 = torch.tensor(laplacian_losses)
        # er3 = torch.stack(test_error)

        ax.plot(er1.detach().cpu(),color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='L_chamfer')
        ax.plot(er2.detach().cpu(),color = 'C1',linestyle='solid',linewidth=lw,alpha=1,label='L_edge')
        ax.plot(er3.detach().cpu(),color = 'C2',linestyle='solid',linewidth=lw,alpha=1,label='L_normal')
        ax.plot(er4.detach().cpu(),color = 'C3',linestyle='solid',linewidth=lw,alpha=1,label='L_lap')
        # ax.plot(check_idx, er3.detach().cpu(),color = 'C2',linestyle='solid',linewidth=lw,alpha=1,label='L_test')
        # ax.set_yscale('log')
        ax.legend()
        fig.savefig(osj(output_path,'label_train_error_epoch{:d}.png'.format(j)),bbox_inches='tight')