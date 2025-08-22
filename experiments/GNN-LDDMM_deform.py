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
from imageseggnn.utils import get_e, create_vtp

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

device = torch.device("cuda:0")
## read in filename list of the data
postprocess_path = "../Data/processed"

file_name_list = os.listdir(osj(postprocess_path,'npys'))
file_name_list = sorted(file_name_list, key=lambda x:float(re.findall("(\d+)",x)[0]))
N = len(file_name_list); print('total num of data:',N)
# print(file_name_list)

# manual selection 
bcptids_all = [torch.tensor([17824,  29000, 27549,1776, 717, 20820], device= device),
                torch.tensor([25395,  33508, 27888,20700, 8123, 2035], device= device),
                ] # you will have to hand pick those point ids from the center of the inlet/outlets

# bcptids_edge_all = [torch.tensor([22702,  27664, 25207, 760, 1442, 18561], device= device),
bcptids_edge_all = [torch.tensor([25012,  28941, 26388, 868, 68, 20655], device= device),
                    torch.tensor([28343,  33397, 29526,19943, 10515, 75], device= device),
                    ] # you will have to hand pick those point ids from the edge of the inlet/outlets

original_path = "../Data/original"
result_path = "./results"

sind = 0 # 0,1 
sids = [63,68]

output_path = osj(result_path,'GNN_LDDMM_id00{:d}_exp1_deform'.format(sids[sind])) # ratio 0.2 lr 1e-3 normal 1  capratio1.2, ablation: no scale
src_filename = [osj(result_path,'GNN_LDDMM_id00{:d}_exp1'.format(sids[sind])),'mesh_3000.ply']

src_filename_copy = osj(output_path,src_filename[-1])
if ose(output_path):
    shutil.rmtree(output_path)

os.mkdir(output_path)
os.mkdir(output_path+'/checkpoints')
os.mkdir(output_path+'/animation')
shutil.copy(osj(*src_filename), src_filename_copy)


i = sind  # change index name

# read in the ct data 
ct_name = os.listdir(osj(original_path,
                file_name_list[i][:-4],
                'Images'))
for ele in ct_name: 
    if 'OSMSC' in ele:
        temp_ct_name = ele
ct = pv.read(osj(original_path,
                    file_name_list[i][:-4],
                    # 'Images',ct_name[0]))
                    'Images',temp_ct_name))

#taking out the gradient in the last channel and added batch channel and channel axis. 
ct_grad_ts = torch.load(osj(postprocess_path,'ct_gradients',file_name_list[i][:-4]+'processed_gradient.pt')).unsqueeze(0)
ct_grad_ts = ct_grad_ts.to(device)
print(ct_grad_ts.shape)

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

# create graph deformation model 
from imageseggnn.mesh_operations.deform_net import *
stages = 3
model = SGCN_deform_s1(3,4,32,2,1/256)
model.to(device)
model.print_parameters()


# load in the source mesh 
src_mesh = Meshes(verts=[verts_hat], faces=[faces])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4 )

e0 = get_e(xyz2hat.denormalize(src_mesh.verts_packed()), ct_grad_ts, xyz2ind) # e0=0.0006   for example 63
label_ply_verts, label_ply_faces = load_ply(osj(result_path,'GNN_LDDMM_id00{:d}_exp1_ref/label_mesh_3000.ply'.format(sids[sind])))
label_ply_verts = label_ply_verts.to(device)
e_label = get_e(label_ply_verts, ct_grad_ts, xyz2ind)


# tolerance = 1e-6
# Number of optimization steps
Niter = 3000
# Weight for the chamfer loss
w_e = 1.0 #0.1    # Weight for mesh edge loss
w_edge = 1.0#  1.0 
# Weight for mesh normal consistency
w_normal = 1.0 #0 #1.0# 0.01
# Weight for mesh laplacian smoothing
w_laplacian = 0.1 #0 #0.1# 0.1

e_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []
total_losses = []

initial_verts = src_mesh.verts_packed()
initial_edges = src_mesh.edges_packed().T
# temp_label = torch.ones_like(initial_verts)*0.01
criterion = nn.MSELoss()

from imageseggnn.modules.my_lddmm import LDDMM
from torch_geometric.nn import fps, knn

number_of_time_pts = 10
indices = fps(verts_hat, ratio=0.2)

# save the indices as a geometry 
pivot_pts = pv.PolyData(xyz2hat.denormalize(initial_verts[indices]).cpu().numpy())
pivot_pts.save(osj(output_path,'mesh_pivot_pts.vtp'))

# manually pick points  
bcptids = bcptids_all[i]
bcptids_edge = bcptids_edge_all[i]
import potpourri3d as pp3d
# convert tensor to numpy 
verts_hat_np, faces_np =verts_hat.detach().cpu().numpy() , faces.detach().cpu().numpy()
solver = pp3d.PointCloudHeatSolver(verts_hat_np)
dists = np.asarray([solver.compute_distance(ele) for ele in bcptids.tolist()])
temp_faces = np.concatenate((np.ones(len(faces))[...,None]*3 , faces.detach().cpu().numpy()), axis = -1).ravel().astype(int)   
temp_mesh = pv.PolyData(xyz2hat.denormalize(verts_hat).detach().cpu().numpy(),faces = temp_faces )
temp_mesh.point_data['diststoinlet'] = dists[0]
temp_mesh.point_data['diststooutlet1'] = dists[1]
temp_mesh.point_data['diststooutlet2'] = dists[2]
temp_mesh.point_data['diststooutlet3'] = dists[3]
temp_mesh.point_data['diststooutlet4'] = dists[4]
temp_mesh.point_data['diststooutlet5'] = dists[5]

def reverse_gaussian_v(x, threshold, kernel_width = 0.1,):
    x = 1.*(x>threshold)*(x-threshold)
    return np.exp(-0.5*((np.array([0.])**2)/kernel_width**2)) - np.exp(-0.5*((x)**2)/kernel_width**2)

bc_pt_radius = dists[np.arange(6),bcptids_edge.tolist()]
gd_ratio = 1.8
bc_pt_cap_ids = [np.nonzero(dists[i] <= gd_ratio*bc_pt_radius[i]) for i in range(6)]
all_pt_scales = np.ones(len(verts_hat))
for i in range(len(bc_pt_cap_ids)):
    all_pt_scales[bc_pt_cap_ids[i]] = reverse_gaussian_v(dists[i][bc_pt_cap_ids[i]], bc_pt_radius[i])

temp_mesh.point_data['scales'] = all_pt_scales
## create the inlet and edge points ##
temp_mesh.point_data['io_centers'] = 0.; temp_mesh.point_data['io_centers'][bcptids.detach().cpu().numpy()] = 1.
temp_mesh.point_data['io_edges'] = 0.; temp_mesh.point_data['io_edges'][bcptids_edge.detach().cpu().numpy()] = 1.
temp_mesh.save(osj(output_path,'scale.vtp'))
all_pt_scales_ts = torch.tensor(all_pt_scales, dtype = torch.float32, device=device).unsqueeze(-1)


model.train()
for j in tqdm(range(Niter+1)):
    ## Deform the mesh

    # Zero your gradients for every batch!
    optimizer.zero_grad()

    deform_vertses_n = model(initial_verts,initial_edges)[indices]
    # normalize the first 3 channels 
    vf = torch.tanh(deform_vertses_n[:,3:]) * deform_vertses_n[:, :3] / (torch.norm(deform_vertses_n[:, :3],dim = -1, keepdim=True)+1e-3)

    my_deform = LDDMM(verts_hat[indices], 2*vf, initial_verts,number_of_time_pts = number_of_time_pts) # ablation no scale
    my_deform.shoot()
    my_deform.flow()
    new_vertses = my_deform.template_pts_t[-1]
    new_src_mesh = Meshes(verts=[new_vertses], faces=[faces])

    if j ==0: 
        save_ply(osj(output_path,'mesh_fit_init.ply'.format(j)), 
                    xyz2hat.denormalize(new_src_mesh.verts_packed()), new_src_mesh.faces_packed())
        
    new_e = get_e(xyz2hat.denormalize(new_src_mesh.verts_packed()), ct_grad_ts, xyz2ind)
    loss_e = -torch.log(new_e) - (-torch.log(e0))
    e_losses.append(loss_e)

    print(loss_e)
    loss_edge = mesh_edge_loss(new_src_mesh)#; print('edge:',loss_edge)
    edge_losses.append(loss_edge)
    # print(loss_edge)

    loss_normal = mesh_normal_consistency(new_src_mesh)#; print('normal:',loss_normal)
    normal_losses.append(loss_normal)
    # print(loss_normal)

    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")#; print('laplacian:',loss_laplacian)
    laplacian_losses.append(loss_laplacian)
    # print(loss_laplacian)
    
    loss = w_e*loss_e + w_edge*loss_edge + w_normal*loss_normal+w_laplacian*loss_laplacian
    loss.backward()
    total_losses.append(loss)
    tqdm.write(str(loss.item()))

    optimizer.step()

    if j % 100 ==0 and j<500:
        temp_mesh = create_vtp(new_src_mesh,xyz2hat, ct_grad_ts,xyz2ind)
        temp_mesh.save(osj(output_path,'mesh_fit_{:d}.vtp'.format(j)))

    if j % 500 ==0:
        torch.save({
                    'epoch': j,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, osj(output_path,'checkpoints','ckpt_{:d}.pt'.format(j)))
    
        temp_mesh = create_vtp(new_src_mesh,xyz2hat, ct_grad_ts,xyz2ind)
        temp_mesh.save(osj(output_path,'mesh_fit_{:d}.vtp'.format(j)))
        print('plotting')

        fig, ax  = plt.subplots(figsize=(10,8))
        lw = 4
        er1 = torch.tensor(e_losses)
        er2 = torch.tensor(edge_losses)
        er3 = torch.tensor(normal_losses)
        er4 = torch.tensor(laplacian_losses)
        er5 = torch.ones_like(er1,device = device) * (-torch.log(e_label) - (-torch.log(e0)))
        er0 = torch.stack(total_losses)


        ax.plot(er0.detach().cpu(),color = 'k',linestyle='solid',linewidth=lw,alpha=1,label='L_total')
        ax.plot(er1.detach().cpu(),color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='L_e')
        ax.plot(er2.detach().cpu(),color = 'C1',linestyle='solid',linewidth=lw,alpha=1,label='L_edge')
        ax.plot(er3.detach().cpu(),color = 'C2',linestyle='solid',linewidth=lw,alpha=1,label='L_normal')
        ax.plot(er4.detach().cpu(),color = 'C3',linestyle='solid',linewidth=lw,alpha=1,label='L_lap')
        ax.plot(er5.detach().cpu(),color = 'C4',linestyle='dashed',linewidth=lw,alpha=1,label='L_e_label')
        # ax.set_yscale('log')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()
        fig.savefig(osj(output_path,'train_fit_error_epoch{:d}.png'.format(j)),bbox_inches='tight')

        torch.save({
            'er0': er0, 
            'er1': er1, 
            'er2': er2, 
            'er3': er3, 
            'er4': er4, 
            'er5': er5, 
        }, osj(output_path,'checkpoints','plots_{:d}.pt'.format(j)))
        # save animation
        animation_mesh = []
        print('saving animations')
        temp_faces = np.concatenate((np.ones(len(faces))[...,None]*3 , faces.detach().cpu().numpy()), axis = -1).ravel().astype(int) 
        for i, template_pts in enumerate(my_deform.template_pts_t):
            # print(temp_faces)
            temp_mesh = pv.PolyData(xyz2hat.denormalize(template_pts).detach().cpu().numpy(),
                                faces = temp_faces )
            temp_mesh.save(osj(output_path,'animation','mesh{:d}.vtp'.format(i)))
            animation_mesh.append(temp_mesh)

            temp_mesh = pv.PolyData(xyz2hat.denormalize(my_deform.control_pts_t[i]).detach().cpu().numpy())
            temp_mesh.point_data['mu'] = my_deform.momenta_t[i].detach().cpu().numpy()
            temp_mesh.save(osj(output_path,'animation','control_points_{:d}.vtp'.format(i)))
        torch.save(torch.stack(my_deform.template_pts_t),osj(output_path,'lddmm_control_points.pt'))
        print('saving animations done')

# CUDA_VISIBLE_DEVICES=1 python GNN-LDDMM_deform.py