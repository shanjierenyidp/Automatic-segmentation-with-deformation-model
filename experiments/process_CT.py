
import os
from os.path import join as osj
import re
import numpy as np
import pyvista as pv
from tqdm import tqdm
import sys
sys.path.insert(1, '../')
from imageseggnn.mesh_operations.transforms import *
from imageseggnn.modules.others import gradient_sobel
import torch

## read in filename list of the data

postprocess_path = "../Data/processed"
## make dirs 
# os.mkdir(osj(postprocess_path,'ct_gradients'))
# os.mkdir(osj(postprocess_path,'ct_gradients'))
# os.mkdir(osj(postprocess_path,'npys_filtered'))
# os.mkdir(osj(postprocess_path,'sptss'))
# os.mkdir(osj(postprocess_path,'vptss'))
# os.mkdir(osj(postprocess_path,'surf_mc'))
# os.mkdir(osj(postprocess_path,'surf_poisson'))

# get the file names in the npys folder 
file_name_list = os.listdir(osj(postprocess_path,'npys'))
# file_name_list = os.listdir(osj(postprocess_path,'npys'))
file_name_list = sorted(file_name_list, key=lambda x:float(re.findall("(\d+)",x)[0]))
N = len(file_name_list); print('total num of data:',N)
# print(file_name_list)

# the original data path 
original_path = "../Data/original"
ct_processed_path = "../Data/processed/CT_input"

# collect all the npy data
npys, vtp_labels, cts, vptss, sptss= [] ,[], [],[],[]

for i in tqdm(range(N)):

    # read in the npyfiles
    print('load npys')
    npy = np.load(osj(postprocess_path,'npys',file_name_list[i]))
    npys.append(npy.copy())

    threshold = 0.5
    print('apply filtering')
    npy_filtered = laplacian(npy, threshold = threshold)
    np.save(osj(postprocess_path,'npys_filtered',file_name_list[i][:-4]+'filtered_{:.2f}.npy'.format(threshold)),npy_filtered)

   # this is the original ct data 
    print('reading in ct names')
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

    # produce ct gradients 
    # read in the image tensors 
    ct_processed = torch.load(osj(ct_processed_path,file_name_list[i][:-4]+'.pt'))
    ct_shape = np.asarray(ct_processed.shape[2:])
    ct_shape_inv = np.flip(ct_shape)
    image = torch.tensor(np.transpose(ct.point_data['ImageScalars'].reshape(ct_shape_inv),(2,1,0)),dtype = torch.float32)
    image_norm = image/torch.max(image)
    # clip values from 0 to 1200; 200 or 0  dont need this becuase processed is from 0-1
    ct_processed = (ct_processed>=0)*ct_processed*(ct_processed<=1200)
    ct.point_data['processed_clamped'] = np.transpose(np.asarray(ct_processed.squeeze()),(2,1,0)).reshape(-1)
    # save the clipped ct to tensor
    print('save preprocessed')
    torch.save(ct_processed,osj(postprocess_path,'ct_gradients',file_name_list[i][:-4]+'.pt'))

    # ct.clear_data()
    ct.save(osj(postprocess_path,'ct_gradients',file_name_list[i][:-4]+'.vti'))
    # obtain the gredient
    image_processed=ct_processed.cpu()[0,0,:,:,:]

    # this is calcualting the orignal raw signal gradient ######################################################
    print('calculate gradient ')
    grad_x, grad_y, grad_z, grad_mag, grad_dir=gradient_sobel(image)
    grad2_x, grad2_y, grad2_z, grad2_mag, grad2_dir=gradient_sobel(grad_mag)
    grads = torch.stack([grad_x, grad_y, grad_z, grad_mag])
    grads2 = torch.stack([grad2_x, grad2_y, grad2_z, grad2_mag])
    grads_grads2 = torch.stack([grads,grads2])
    torch.save(grads_grads2,osj(postprocess_path,'ct_gradients',file_name_list[i][:-4]+'_grads_grads2.pt'))
    ct_copy = ct.copy()
    ct_copy.clear_data()
    ct_copy.point_data['grads'] = np.transpose(np.asarray(grad_mag),(2,1,0)).reshape(-1)
    ct_copy.save(osj(postprocess_path,'ct_gradients',file_name_list[i][:-4]+'gradient.vti'))
    # ct.point_data['grads2'] = np.transpose(np.asarray(grad2_mag),(2,1,0)).reshape(-1)
    ############################################################################################################

    # # gradient of the preprocessed signal (thresholded and normed) ############################################################################
    grad_x, grad_y, grad_z, grad_mag, grad_dir=gradient_sobel(image_processed)
    grad2_x, grad2_y, grad2_z, grad2_mag, grad2_dir=gradient_sobel(grad_mag)
    grads = torch.cat([grad_x.unsqueeze(-1), grad_y.unsqueeze(-1), grad_z.unsqueeze(-1), grad_mag.unsqueeze(-1)], dim = -1)
    ct_copy = ct.copy()
    ct_copy.clear_data()
    # ct.point_data['processed'] = np.transpose(np.asarray(ct_processed.squeeze()),(2,1,0)).reshape(-1)
    ct_copy.point_data['processed_grads'] = np.transpose(np.asarray(grad_mag),(2,1,0)).reshape(-1)
    # ct.point_data['processed_grads2'] = np.transpose(np.asarray(grad2_mag),(2,1,0)).reshape(-1)
    ct_copy.save(osj(postprocess_path,'ct_gradients',file_name_list[i][:-4]+'processed_gradient.vti'))
    torch.save(grads,osj(postprocess_path,'ct_gradients',file_name_list[i][:-4]+'processed_gradient.pt'))
    ###########################################################################################################


    grad_x, grad_y, grad_z, grad_mag, grad_dir=gradient_sobel(image_norm)
    grad2_x, grad2_y, grad2_z, grad2_mag, grad2_dir=gradient_sobel(grad_mag)
    grads = torch.stack([grad_x, grad_y, grad_z, grad_mag])

    print(ct_processed.shape) # this shape is [1,1,xx,yy,zz]
    print(grad_x.shape,image_norm.shape)
    torch.save(grad_mag,osj(postprocess_path,'ct_gradients',file_name_list[i][:-4]+'_norm.pt'))
    grads2 = torch.stack([grad2_x, grad2_y, grad2_z, grad2_mag])
    grads_grads2 = torch.stack([grads,grads2])
    ct_copy = ct.copy()
    ct_copy.clear_data()
    ct_copy.point_data['norm_imagescalar'] = np.transpose(np.asarray(image_norm),(2,1,0)).reshape(-1)
    ct_copy.point_data['norm_grads'] = np.transpose(np.asarray(grad_mag),(2,1,0)).reshape(-1)
    ct_copy.point_data['norm_grads2'] = np.transpose(np.asarray(grad2_mag),(2,1,0)).reshape(-1)
    ct_copy.save(osj(postprocess_path,'ct_gradients',file_name_list[i][:-4]+'norm.vti'))

    ct_copy = ct.copy()
    ct_copy.clear_data()
    ct_copy.point_data['prediction'] = np.transpose(npy,(2,1,0)).reshape(-1)
    ct_copy.save(osj(postprocess_path,'ct_gradients',file_name_list[i][:-4]+'_prediction.vti'))

    # pad the npy_filter by one in case the vessel is at tensor boundary 
    npy_filtered_padded = np.zeros((npy_filtered.shape[0]+2,npy_filtered.shape[1]+2,npy_filtered.shape[2]+2))
    npy_filtered_padded[1:-1, 1:-1, 1:-1] = npy_filtered

    # convert npy to point cloud in physical space 
    volume_pts = get_volume_pos(npy, ct)
    # volume_pts = get_volume_pos(npy_filtered, ct)
    temp = pv.PolyData(volume_pts)    
    # temp_file_name = osj(postprocess_path,'vptss',file_name_list[i][:-4]+'_vpts.vtp')
    temp_file_name = osj(postprocess_path,'vptss',file_name_list[i][:-4]+'_vpts_org.vtp')
    temp.save(temp_file_name)

    # get shell points from volumen points
    shell_pts = get_shell_pos(npy, ct)
    # shell_pts = get_shell_pos(npy_filtered, ct)
    temp = pv.PolyData(shell_pts)
    # temp_file_name = osj(postprocess_path,'sptss',file_name_list[i][:-4]+'_spts.vtp')
    temp_file_name = osj(postprocess_path,'sptss',file_name_list[i][:-4]+'_spts_org.vtp')
    temp.save(temp_file_name)

    # reconstruct using marching cube 
    mesh_ind = get_isosurf(npy,iso_value = 0.5)
    mesh_ind = get_isosurf(npy_filtered,iso_value = 0.5)

    mesh_phy = mesh_ind.copy()
    mesh_phy.points = ind2phy_fast(mesh_ind.points,ct)
    mesh_phy.save(osj(postprocess_path,'surf_mc',file_name_list[i][:-4]+'surf_filtered_mc.vtp'))
    mesh_phy.save(osj(postprocess_path,'surf_mc',file_name_list[i][:-4]+'surf_filtered_mc.ply'))

    # remesh the mc using pyacvd
    remesh = ACVD(mesh_phy, keep_ratio = 1)
    remesh.save(osj(postprocess_path,'surf_mc',file_name_list[i][:-4]+'surf_filtered_mc_remesh.vtp'))
    remesh.save(osj(postprocess_path,'surf_mc',file_name_list[i][:-4]+'surf_filtered_mc_remesh.ply'))


print('Done') 
