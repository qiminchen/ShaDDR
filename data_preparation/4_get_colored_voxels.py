import os
import numpy as np
import cv2
import binvox_rw_customized
import argparse
import time
import h5py
from skimage import measure
import get_point_cloud
import cutils
from sklearn.neighbors import KDTree
from scipy.ndimage import binary_erosion, binary_dilation

parser = argparse.ArgumentParser()
parser.add_argument("class_id", type=str, help="shapenet category id")
parser.add_argument("share_id", type=int, help="id of the share [0]")
parser.add_argument("share_total", type=int, help="total num of shares [1]")
FLAGS = parser.parse_args()

class_id = FLAGS.class_id
target_dir = "./" + class_id + "/"
if not os.path.exists(target_dir):
    print("ERROR: this dir does not exist: " + target_dir)
    exit(-1)

share_id = FLAGS.share_id
share_total = FLAGS.share_total

obj_names = os.listdir(target_dir)
obj_names = sorted(obj_names)

obj_names_ = []
for i in range(len(obj_names)):
    if i % share_total == share_id:
        obj_names_.append(obj_names[i])
obj_names = obj_names_

voxel_size = 512

for idx in range(len(obj_names)):
    this_name = target_dir + obj_names[idx] + "/model_depth_fusion.binvox"
    out_name = target_dir + obj_names[idx] + "/voxel_color.hdf5"
    print(idx, len(obj_names), this_name)
    batch_voxels = binvox_rw_customized.read_voxels(this_name)
    
    #get approximate normal direction
    #0 = x+ direction
    #1 = x- direction
    #2 = y+ direction
    #3 = y- direction
    #4 = z+ direction
    #5 = z- direction
    voxel_normals = np.zeros([voxel_size,voxel_size,voxel_size], np.uint8)
    cutils.get_voxel_approximate_normal_direction(batch_voxels,voxel_normals)

    #record which voxels are surface voxels
    voxel_surface = np.zeros([voxel_size,voxel_size,voxel_size], np.uint8)
    cutils.get_voxel_surface_flag(batch_voxels,voxel_surface)

    #output
    voxel_color = np.zeros([voxel_size,voxel_size,voxel_size,4], np.uint8)
    voxel_color[:, :, :, 3] = batch_voxels

    #record which voxels have been assigned colors
    voxel_filled_flag = np.zeros([voxel_size,voxel_size,voxel_size], np.uint8)

    # --- baking start ---

    #y+
    mask = (np.logical_not(voxel_filled_flag) | (voxel_normals==2)) & voxel_surface

    t = 0
    img_in = cv2.imread(target_dir + obj_names[idx] + "/"+str(t)+".png", cv2.IMREAD_UNCHANGED)
    idxs = 511-np.argmax(batch_voxels[:,::-1,:],1)
    img_in = np.transpose(img_in, [1,0,2])
    for i in range(512):
        for j in range(512):
            tmask = mask[i,idxs[i,j],j,None]
            voxel_color[i,idxs[i,j],j,:3] = img_in[i,j,:3]*tmask + voxel_color[i,idxs[i,j],j,:3]*(1-tmask)
            voxel_filled_flag[i,idxs[i,j],j] = 1

    #y-
    mask = (np.logical_not(voxel_filled_flag) | (voxel_normals==3)) & voxel_surface

    t = 1
    img_in = cv2.imread(target_dir + obj_names[idx] + "/"+str(t)+".png", cv2.IMREAD_UNCHANGED)
    idxs = np.argmax(batch_voxels[:,:,:],1)
    img_in = np.transpose(img_in[::-1], [1,0,2])
    for i in range(512):
        for j in range(512):
            tmask = mask[i,idxs[i,j],j,None]
            voxel_color[i,idxs[i,j],j,:3] = img_in[i,j,:3]*tmask + voxel_color[i,idxs[i,j],j,:3]*(1-tmask)
            voxel_filled_flag[i,idxs[i,j],j] = 1

    #z+
    mask = (np.logical_not(voxel_filled_flag) | (voxel_normals==4)) & voxel_surface

    t = 2
    img_in = cv2.imread(target_dir + obj_names[idx] + "/"+str(t)+".png", cv2.IMREAD_UNCHANGED)
    idxs = 511-np.argmax(batch_voxels[:,:,::-1],2)
    img_in = np.transpose(img_in[::-1], [1,0,2])
    for i in range(512):
        for j in range(512):
            tmask = mask[i,j,idxs[i,j],None]
            voxel_color[i,j,idxs[i,j],:3] = img_in[i,j,:3]*tmask + voxel_color[i,j,idxs[i,j],:3]*(1-tmask)
            voxel_filled_flag[i,j,idxs[i,j]] = 1

    #z-
    mask = (np.logical_not(voxel_filled_flag) | (voxel_normals==5)) & voxel_surface

    t = 3
    img_in = cv2.imread(target_dir + obj_names[idx] + "/"+str(t)+".png", cv2.IMREAD_UNCHANGED)
    idxs = np.argmax(batch_voxels[:,:,:],2)
    img_in = np.transpose(img_in[::-1,::-1], [1,0,2])
    for i in range(512):
        for j in range(512):
            tmask = mask[i,j,idxs[i,j],None]
            voxel_color[i,j,idxs[i,j],:3] = img_in[i,j,:3]*tmask + voxel_color[i,j,idxs[i,j],:3]*(1-tmask)
            voxel_filled_flag[i,j,idxs[i,j]] = 1

    #x+
    mask = (np.logical_not(voxel_filled_flag) | (voxel_normals==0)) & voxel_surface

    t = 4
    img_in = cv2.imread(target_dir + obj_names[idx] + "/"+str(t)+".png", cv2.IMREAD_UNCHANGED)
    idxs = 511-np.argmax(batch_voxels[::-1,:,:],0)
    img_in = img_in[::-1,::-1]
    for i in range(512):
        for j in range(512):
            tmask = mask[idxs[i,j],i,j,None]
            voxel_color[idxs[i,j],i,j,:3] = img_in[i,j,:3]*tmask + voxel_color[idxs[i,j],i,j,:3]*(1-tmask)
            voxel_filled_flag[idxs[i,j],i,j] = 1

    #x-
    mask = (np.logical_not(voxel_filled_flag) | (voxel_normals==1)) & voxel_surface

    t = 5
    img_in = cv2.imread(target_dir + obj_names[idx] + "/"+str(t)+".png", cv2.IMREAD_UNCHANGED)
    idxs = np.argmax(batch_voxels[:,:,:],0)
    img_in = img_in[::-1]
    for i in range(512):
        for j in range(512):
            tmask = mask[idxs[i,j],i,j,None]
            voxel_color[idxs[i,j],i,j,:3] = img_in[i,j,:3]*tmask + voxel_color[idxs[i,j],i,j,:3]*(1-tmask)
            voxel_filled_flag[idxs[i,j],i,j] = 1

    # --- baking end ---

    #inpaint colors of voxels
    voxel_filled_flag = voxel_filled_flag & voxel_surface
    cutils.inpaint_surface(voxel_color,voxel_surface,voxel_filled_flag)
    
    filled_coord = np.transpose(np.nonzero(voxel_filled_flag>0)).astype(np.int32)
    filled_color = voxel_color[filled_coord[:,0],filled_coord[:,1],filled_coord[:,2],:3]

    #build kdtree with surface voxels
    kd_tree = KDTree(filled_coord, leaf_size=8)

    # inpaint other voxels, extremely slow
    tofill_coord = np.transpose(np.nonzero(voxel_filled_flag==0)).astype(np.int32)
    closest_idx = kd_tree.query(tofill_coord, k=1, return_distance=False)
    closest_idx = np.reshape(closest_idx,[-1])
    voxel_color[tofill_coord[:,0],tofill_coord[:,1],tofill_coord[:,2],:3] = filled_color[closest_idx]

    '''
    # test, get colored mesh
    verts, faces, _, _ = measure.marching_cubes(0.5-voxel_color[:,:,:,3], 0)
    verts_int1 = (verts).astype(np.int32)
    verts_int2 = (verts+0.5).astype(np.int32)
    verts = (verts+0.5)/voxel_size-0.5
    vcolors = np.maximum(voxel_color[verts_int1[:,0],verts_int1[:,1],verts_int1[:,2]],voxel_color[verts_int2[:,0],verts_int2[:,1],verts_int2[:,2]])
    vcolors = vcolors.astype(np.uint8)
    get_point_cloud.write_ply_triangle_color(out_name+".surfacevoxel"+str(voxel_size)+".ply", verts, vcolors[:,2::-1], faces)
    '''

    hdf5_file = h5py.File(out_name, 'w')
    hdf5_file.create_dataset("voxel_color", [voxel_size,voxel_size,voxel_size,4], np.uint8, compression=9)
    hdf5_file["voxel_color"][:] = voxel_color
    hdf5_file.close()
