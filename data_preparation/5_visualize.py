import numpy as np
import os
import argparse
import time
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("class_id", type=str, help="shapenet category id")
FLAGS = parser.parse_args()


def write_ply_triangle_color(name, vertices, colors, triangles):
    fout = open(name, 'w')
    fout.write( "ply\n" +
                "format ascii 1.0\n" +
                "element vertex "+str(len(vertices))+"\n" +
                "property float x\n" +
                "property float y\n" +
                "property float z\n" +
                "property uchar red\n" +
                "property uchar green\n" +
                "property uchar blue\n" +
                "element face "+str(len(triangles))+"\n" +
                "property list uchar int vertex_index\n" +
                "end_header\n")
    for i in range(len(vertices)):
        fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+
                   str(int(colors[i,0]))+" "+str(int(colors[i,1]))+" "+str(int(colors[i,2]))+"\n")
    for i in range(len(triangles)):
        fout.write("3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
    fout.close()


class_id = FLAGS.class_id
target_dir = "./" + class_id + "/"
if not os.path.exists(target_dir):
    print("ERROR: this dir does not exist: " + target_dir)
    exit(-1)

obj_names = os.listdir(target_dir)
obj_names = sorted(obj_names)

for idx in range(len(obj_names)):
    vox_dir = target_dir + obj_names[idx] + "/model_depth_fusion.binvox"
    voxel_model_file = open(vox_dir, 'rb')
    voxel_model = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=True).data.astype(np.uint8)
    voxel_size = voxel_model.shape[0]

    data_dict = h5py.File(target_dir + obj_names[idx] + "/voxel_color.hdf5", 'r')
    tmp_color_raw = data_dict["voxel_color"][:]
    data_dict.close()
    voxel_color = tmp_color_raw[:, :, :, :3]
    voxel_color = voxel_color[:, :, :, [2, 1, 0]]

    vertices, faces = mcubes.marching_cubes(voxel_model, 0.4)
    vertices_int1 = vertices.astype(np.int32)
    vertices_int2 = (vertices + 0.5).astype(np.int32)
    vertices = (vertices + 0.5) / voxel_size - 0.5

    vertices_colors = np.maximum(voxel_color[vertices_int1[:, 0], vertices_int1[:, 1], vertices_int1[:, 2]],
                                 voxel_color[vertices_int2[:, 0], vertices_int2[:, 1], vertices_int2[:, 2]])
    vertices_colors = vertices_colors.astype(np.uint8)
    write_ply_triangle_color(target_dir + obj_names[idx] + "/color_mesh.ply", vertices, vertices_colors, faces)
