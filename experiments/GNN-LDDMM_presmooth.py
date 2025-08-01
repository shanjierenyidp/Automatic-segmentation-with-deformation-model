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
from torch import nn 

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
result_path = "./results/experiement1"
sind = 0 # 0,1 
sids = [63,68]
sidstrs = ['0063_1001','0068_0001']

output_path = osj(result_path,'GNN_LDDMM_id00{:d}_exp1'.format(sids[sind])) # specify results path 
if ose(output_path):
    shutil.rmtree(output_path)

os.mkdir(output_path)
#### start fitting ####

src_filename = [postprocess_path+'/surf_mc',sidstrs[sind]+'surf_mc_remesh.ply']
mesh_temp = pv.read(osj(src_filename[0], src_filename[1][:-4]+'.vtp'))
mesh_temp.save(osj(*src_filename))
src_filename_copy = osj(output_path,src_filename[-1])
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
# print('verify normalizer is correct: ',torch.max(abs(verts)), torch.max(abs(xyz2hat.denormalize(verts_hat)- verts)))

# load in the source mesh 
src_mesh = Meshes(verts=[verts_hat], faces=[faces])
deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
# optimizer = torch.optim.SGD([deform_verts], lr=1e-3, momentum=0.9)
optimizer = torch.optim.Adam([deform_verts], lr=1e-3, weight_decay = 1e-4)

e0 = get_e(xyz2hat.denormalize(src_mesh.verts_packed()), ct_grad_ts, xyz2ind) # e0=0.0006   for example 63

Niter = 3000
w_e = 0.1
w_edge = 1.0#  1.0 
w_normal = 0.1# 0.01
w_laplacian = 0.1# 0.1

e_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []
total_losses = []

initial_verts = src_mesh.verts_packed()
initial_edges = src_mesh.edges_packed().T
criterion = nn.MSELoss()


for j in tqdm(range(Niter+1)): # iteratively smooth the surface 
    optimizer.zero_grad()
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    if j ==0: 
        save_ply(osj(output_path,'mesh_init.ply'.format(j)), 
                    xyz2hat.denormalize(new_src_mesh.verts_packed()), new_src_mesh.faces_packed())

    new_e = get_e(xyz2hat.denormalize(new_src_mesh.verts_packed()))
    loss_e = -torch.log(new_e) - (-torch.log(e0))
    e_losses.append(loss_e.item())
        # print(loss_e)
    loss_edge = mesh_edge_loss(new_src_mesh)#; print('edge:',loss_edge)
    edge_losses.append(loss_edge.item())

    loss_normal = mesh_normal_consistency(new_src_mesh)#; print('normal:',loss_normal)
    normal_losses.append(loss_normal.item())

    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")#; print('laplacian:',loss_laplacian)
    laplacian_losses.append(loss_laplacian.item())

    loss = w_edge*loss_edge + w_normal*loss_normal+w_laplacian*loss_laplacian
    loss.backward()
    optimizer.step()
    total_losses.append(loss.item())
    tqdm.write(str(loss.item()))

    if j % 10==0 and j<100:
        
        save_ply(osj(output_path,'mesh_{:d}.ply'.format(j)), 
                    xyz2hat.denormalize(new_src_mesh.verts_packed()), new_src_mesh.faces_packed())
        
    if j % 500 ==0:
        
        save_ply(osj(output_path,'mesh_{:d}.ply'.format(j)), 
                    xyz2hat.denormalize(new_src_mesh.verts_packed()), new_src_mesh.faces_packed())
        
        # print('plotting')

        fig, ax = plt.subplots(figsize=(10,8))
        lw = 4
        er1 = torch.tensor(e_losses)
        er2 = torch.tensor(edge_losses)
        er3 = torch.tensor(normal_losses)
        er4 = torch.tensor(laplacian_losses)
        er0 = torch.tensor(total_losses)


        ax.plot(er0,color = 'k',linestyle='solid',linewidth=lw,alpha=1,label='L_total')
        ax.plot(er1,color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='L_e')
        ax.plot(er2,color = 'C1',linestyle='solid',linewidth=lw,alpha=1,label='L_edge')
        ax.plot(er3,color = 'C2',linestyle='solid',linewidth=lw,alpha=1,label='L_normal')
        ax.plot(er4,color = 'C3',linestyle='solid',linewidth=lw,alpha=1,label='L_lap')
        # ax.set_yscale('log')
        ax.legend()
        fig.savefig(osj(output_path,'train_error_epoch{:d}.png'.format(j)),bbox_inches='tight')
        torch.save(torch.stack((er0,er1,er2,er3,er4)), osj(output_path,'erros_epoch{:d}.pt'.format(j)))