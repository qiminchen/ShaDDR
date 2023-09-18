import numpy as np
import os
import binvox_rw_customized
import cutils
import argparse
import time

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


cube_sizex = 16
cube_sizey = 16
cube_sizez = 40
cube_size2 = 2

for i in range(len(obj_names)):
    this_name = target_dir + obj_names[i] + "/model.binvox"
    out_name = target_dir + obj_names[i] + "/model_depth_fusion.binvox"
    print(i,len(obj_names),this_name)

    batch_voxels = binvox_rw_customized.read_voxels(this_name,fix_coords=False)

    rendering = np.full([5120, 5120, 6], 65536, np.int32)
    cutils.depth_fusion_XZY_5views(batch_voxels, rendering)
    del rendering

    state_ctr = np.zeros([512 * 512 * 64, 2], np.int32)
    cutils.get_state_ctr(batch_voxels,state_ctr)
    binvox_rw_customized.write(out_name, batch_voxels.shape, state_ctr)
    del state_ctr
    del batch_voxels
