import os
import numpy as np
import argparse

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


def normalize_obj(this_name, out_name):
    fin = open(this_name,'r')
    lines = fin.readlines()
    fin.close()
    
    # read shape
    vertices = []
    triangles = []
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) == 0:
            continue
        elif line[0] == 'v':
            vertices.append([float(line[1]), float(line[2]), float(line[3])])
        elif line[0] == 'f':
            triangles.append([int(line[1].split("/")[0]), int(line[2].split("/")[0]), int(line[3].split("/")[0])])
    vertices = np.array(vertices, np.float32)
    triangles = np.array(triangles, np.int32)-1
    
    #remove isolated points
    vertices_used_flag = np.full([len(vertices)], 0, np.int32)
    vertices_used_flag[np.reshape(triangles,[-1])] = 1
    vertices_ = vertices[vertices_used_flag>0]

    #normalize max=1
    x_max, y_max, z_max = np.max(vertices_, 0)
    x_min, y_min, z_min = np.min(vertices_, 0)
    x_mid, y_mid, z_mid = (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2
    x_scale, y_scale, z_scale = x_max - x_min, y_max - y_min, z_max - z_min
    #scale = max(x_scale,y_scale,z_scale)
    scale = np.sqrt(x_scale * x_scale + y_scale * y_scale + z_scale * z_scale)

    #write normalized shape
    fout = open(out_name, 'w')
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) == 0:
            continue
        elif line[0] == 'v':
            x = (float(line[1]) - x_mid) / scale
            y = (float(line[2]) - y_mid) / scale
            z = (float(line[3]) - z_mid) / scale
            fout.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")
        else:
            fout.write(lines[i])
    fout.close()


for i in range(len(obj_names)):
    this_name = target_dir + obj_names[i] + "/model.obj"
    out_name = target_dir + obj_names[i] + "/model_normalized.obj"
    print(i,len(obj_names),this_name)
    normalize_obj(this_name,out_name)
