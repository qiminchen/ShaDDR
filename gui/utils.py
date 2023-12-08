import numpy as np
import binvox_rw


def get_vox_from_binvox(objname):
    # get voxel models
    voxel_model_file = open(objname, 'rb')
    voxel_model_512 = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=True).data.astype(np.uint8)
    step_size = 2
    voxel_model_256 = voxel_model_512[0::step_size, 0::step_size, 0::step_size]
    for i in range(step_size):
        for j in range(step_size):
            for k in range(step_size):
                voxel_model_256 = np.maximum(voxel_model_256, voxel_model_512[i::step_size, j::step_size, k::step_size])
    # add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
    # voxel_model_256 = np.flip(np.transpose(voxel_model_256, (2,1,0)),2)

    return voxel_model_256


def get_vox_from_binvox_1over2(objname):
    # get voxel models
    voxel_model_file = open(objname, 'rb')
    voxel_model_512 = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=True).data.astype(np.uint8)
    step_size = 4
    padding_size = 256 % step_size
    output_padding = 128 - (256 // step_size)
    # voxel_model_512 = voxel_model_512[padding_size:-padding_size,padding_size:-padding_size,padding_size:-padding_size]
    voxel_model_128 = voxel_model_512[0::step_size, 0::step_size, 0::step_size]
    for i in range(step_size):
        for j in range(step_size):
            for k in range(step_size):
                voxel_model_128 = np.maximum(voxel_model_128, voxel_model_512[i::step_size, j::step_size, k::step_size])
    # add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
    # voxel_model_128 = np.flip(np.transpose(voxel_model_128, (2,1,0)),2)
    voxel_model_256 = np.zeros([256, 256, 256], np.uint8)
    voxel_model_256[output_padding:-output_padding, output_padding:-output_padding, output_padding:-output_padding] = voxel_model_128

    return voxel_model_256


def get_vox_from_binvox_1over2_return_small(objname):
    # get voxel models
    voxel_model_file = open(objname, 'rb')
    voxel_model_512 = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=True).data.astype(np.uint8)
    step_size = 4
    padding_size = 256 % step_size
    output_padding = 128 - (256 // step_size)
    # voxel_model_512 = voxel_model_512[padding_size:-padding_size,padding_size:-padding_size,padding_size:-padding_size]
    voxel_model_128 = voxel_model_512[0::step_size, 0::step_size, 0::step_size]
    for i in range(step_size):
        for j in range(step_size):
            for k in range(step_size):
                voxel_model_128 = np.maximum(voxel_model_128, voxel_model_512[i::step_size, j::step_size, k::step_size])
    # add flip & transpose to convert coord from shapenet_v1 to shapenet_v2
    # voxel_model_128 = np.flip(np.transpose(voxel_model_128, (2,1,0)),2)

    return voxel_model_128


def write_ply_point(name, vertices):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    fout.close()


def write_ply_point_normal(name, vertices, normals=None):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property float nx\n")
    fout.write("property float ny\n")
    fout.write("property float nz\n")
    fout.write("end_header\n")
    if normals is None:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + " " + str(vertices[ii, 3]) + " " + str(
                vertices[ii, 4]) + " " + str(vertices[ii, 5]) + "\n")
    else:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + " " + str(normals[ii, 0]) + " " + str(
                normals[ii, 1]) + " " + str(normals[ii, 2]) + "\n")
    fout.close()


def write_ply_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face " + str(len(triangles)) + "\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    for ii in range(len(triangles)):
        fout.write("3 " + str(triangles[ii, 0]) + " " + str(triangles[ii, 1]) + " " + str(triangles[ii, 2]) + "\n")
    fout.close()


def write_obj_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    for ii in range(len(vertices)):
        fout.write("v " + str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    for ii in range(len(triangles)):
        fout.write("f " + str(int(triangles[ii, 0] + 1)) + " " + str(int(triangles[ii, 1] + 1)) + " " + str(int(triangles[ii, 2] + 1)) + "\n")
    fout.close()


def get_simple_coarse_voxel_part_color(voxels, segmentations, color_maps):
    all_vertices_len = 0
    all_triangles_len = 0
    all_vertices_max = 128 ** 3
    all_triangles_max = 128 ** 3
    all_vertices = np.zeros([all_vertices_max, 3], np.float32)
    all_colors = np.zeros([all_vertices_max, 3], np.float32)
    all_normals = np.zeros([all_triangles_max, 3], np.float32)
    all_triangles = np.zeros([all_triangles_max, 3], np.int32)

    dimx, dimy, dimz = voxels.shape

    for i in range(0, dimx):
        for j in range(0, dimy):
            for k in range(0, dimz):
                if voxels[i, j, k] == 0: continue

                # x negative
                if i > 0 and voxels[i - 1, j, k] == 0 or i <= 0:

                    # vertices
                    all_vertices[all_vertices_len, 0] = i
                    all_vertices[all_vertices_len, 1] = j
                    all_vertices[all_vertices_len, 2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i
                    all_vertices[all_vertices_len, 1] = j + 1
                    all_vertices[all_vertices_len, 2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i
                    all_vertices[all_vertices_len, 1] = j + 1
                    all_vertices[all_vertices_len, 2] = k + 1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i
                    all_vertices[all_vertices_len, 1] = j
                    all_vertices[all_vertices_len, 2] = k + 1
                    all_vertices_len += 1

                    # triangles
                    all_triangles[all_triangles_len, 0] = all_vertices_len - 4
                    all_triangles[all_triangles_len, 1] = all_vertices_len - 2
                    all_triangles[all_triangles_len, 2] = all_vertices_len - 3
                    all_normals[all_triangles_len, 0] = -1.0
                    all_normals[all_triangles_len, 1] = 0.0
                    all_normals[all_triangles_len, 2] = 0.0
                    all_colors[all_triangles_len, :] = color_maps[segmentations[i, j, k]]
                    all_triangles_len += 1
                    all_triangles[all_triangles_len, 0] = all_vertices_len - 4
                    all_triangles[all_triangles_len, 1] = all_vertices_len - 1
                    all_triangles[all_triangles_len, 2] = all_vertices_len - 2
                    all_normals[all_triangles_len, 0] = -1.0
                    all_normals[all_triangles_len, 1] = 0.0
                    all_normals[all_triangles_len, 2] = 0.0
                    all_colors[all_triangles_len, :] = color_maps[segmentations[i, j, k]]
                    all_triangles_len += 1

                # x positive
                if i + 1 < dimx and voxels[i + 1, j, k] == 0 or i + 1 >= dimx:
                    # vertices
                    all_vertices[all_vertices_len, 0] = i + 1
                    all_vertices[all_vertices_len, 1] = j
                    all_vertices[all_vertices_len, 2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i + 1
                    all_vertices[all_vertices_len, 1] = j + 1
                    all_vertices[all_vertices_len, 2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i + 1
                    all_vertices[all_vertices_len, 1] = j + 1
                    all_vertices[all_vertices_len, 2] = k + 1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i + 1
                    all_vertices[all_vertices_len, 1] = j
                    all_vertices[all_vertices_len, 2] = k + 1
                    all_vertices_len += 1

                    # triangles
                    all_triangles[all_triangles_len, 0] = all_vertices_len - 4
                    all_triangles[all_triangles_len, 1] = all_vertices_len - 3
                    all_triangles[all_triangles_len, 2] = all_vertices_len - 2
                    all_normals[all_triangles_len, 0] = 1.0
                    all_normals[all_triangles_len, 1] = 0.0
                    all_normals[all_triangles_len, 2] = 0.0
                    all_colors[all_triangles_len, :] = color_maps[segmentations[i, j, k]]
                    all_triangles_len += 1
                    all_triangles[all_triangles_len, 0] = all_vertices_len - 4
                    all_triangles[all_triangles_len, 1] = all_vertices_len - 2
                    all_triangles[all_triangles_len, 2] = all_vertices_len - 1
                    all_normals[all_triangles_len, 0] = 1.0
                    all_normals[all_triangles_len, 1] = 0.0
                    all_normals[all_triangles_len, 2] = 0.0
                    all_colors[all_triangles_len, :] = color_maps[segmentations[i, j, k]]
                    all_triangles_len += 1

                # y negative
                if j > 0 and voxels[i, j - 1, k] == 0 or j <= 0:
                    # vertices
                    all_vertices[all_vertices_len, 0] = i
                    all_vertices[all_vertices_len, 1] = j
                    all_vertices[all_vertices_len, 2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i + 1
                    all_vertices[all_vertices_len, 1] = j
                    all_vertices[all_vertices_len, 2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i + 1
                    all_vertices[all_vertices_len, 1] = j
                    all_vertices[all_vertices_len, 2] = k + 1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i
                    all_vertices[all_vertices_len, 1] = j
                    all_vertices[all_vertices_len, 2] = k + 1
                    all_vertices_len += 1

                    # triangles
                    all_triangles[all_triangles_len, 0] = all_vertices_len - 4
                    all_triangles[all_triangles_len, 1] = all_vertices_len - 3
                    all_triangles[all_triangles_len, 2] = all_vertices_len - 2
                    all_normals[all_triangles_len, 0] = 0.0
                    all_normals[all_triangles_len, 1] = -1.0
                    all_normals[all_triangles_len, 2] = 0.0
                    all_colors[all_triangles_len, :] = color_maps[segmentations[i, j, k]]
                    all_triangles_len += 1
                    all_triangles[all_triangles_len, 0] = all_vertices_len - 4
                    all_triangles[all_triangles_len, 1] = all_vertices_len - 2
                    all_triangles[all_triangles_len, 2] = all_vertices_len - 1
                    all_normals[all_triangles_len, 0] = 0.0
                    all_normals[all_triangles_len, 1] = -1.0
                    all_normals[all_triangles_len, 2] = 0.0
                    all_colors[all_triangles_len, :] = color_maps[segmentations[i, j, k]]
                    all_triangles_len += 1

                # y positive
                if j + 1 < dimy and voxels[i, j + 1, k] == 0 or j + 1 >= dimy:
                    # vertices
                    all_vertices[all_vertices_len, 0] = i
                    all_vertices[all_vertices_len, 1] = j + 1
                    all_vertices[all_vertices_len, 2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i + 1
                    all_vertices[all_vertices_len, 1] = j + 1
                    all_vertices[all_vertices_len, 2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i + 1
                    all_vertices[all_vertices_len, 1] = j + 1
                    all_vertices[all_vertices_len, 2] = k + 1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i
                    all_vertices[all_vertices_len, 1] = j + 1
                    all_vertices[all_vertices_len, 2] = k + 1
                    all_vertices_len += 1

                    # triangles
                    all_triangles[all_triangles_len, 0] = all_vertices_len - 4
                    all_triangles[all_triangles_len, 1] = all_vertices_len - 2
                    all_triangles[all_triangles_len, 2] = all_vertices_len - 3
                    all_normals[all_triangles_len, 0] = 0.0
                    all_normals[all_triangles_len, 1] = 1.0
                    all_normals[all_triangles_len, 2] = 0.0
                    all_colors[all_triangles_len, :] = color_maps[segmentations[i, j, k]]
                    all_triangles_len += 1
                    all_triangles[all_triangles_len, 0] = all_vertices_len - 4
                    all_triangles[all_triangles_len, 1] = all_vertices_len - 1
                    all_triangles[all_triangles_len, 2] = all_vertices_len - 2
                    all_normals[all_triangles_len, 0] = 0.0
                    all_normals[all_triangles_len, 1] = 1.0
                    all_normals[all_triangles_len, 2] = 0.0
                    all_colors[all_triangles_len, :] = color_maps[segmentations[i, j, k]]
                    all_triangles_len += 1

                # z negative
                if k > 0 and voxels[i, j, k - 1] == 0 or k <= 0:
                    # vertices
                    all_vertices[all_vertices_len, 0] = i
                    all_vertices[all_vertices_len, 1] = j
                    all_vertices[all_vertices_len, 2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i + 1
                    all_vertices[all_vertices_len, 1] = j
                    all_vertices[all_vertices_len, 2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i + 1
                    all_vertices[all_vertices_len, 1] = j + 1
                    all_vertices[all_vertices_len, 2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i
                    all_vertices[all_vertices_len, 1] = j + 1
                    all_vertices[all_vertices_len, 2] = k
                    all_vertices_len += 1

                    # triangles
                    all_triangles[all_triangles_len, 0] = all_vertices_len - 4
                    all_triangles[all_triangles_len, 1] = all_vertices_len - 2
                    all_triangles[all_triangles_len, 2] = all_vertices_len - 3
                    all_normals[all_triangles_len, 0] = 0.0
                    all_normals[all_triangles_len, 1] = 0.0
                    all_normals[all_triangles_len, 2] = -1.0
                    all_colors[all_triangles_len, :] = color_maps[segmentations[i, j, k]]
                    all_triangles_len += 1
                    all_triangles[all_triangles_len, 0] = all_vertices_len - 4
                    all_triangles[all_triangles_len, 1] = all_vertices_len - 1
                    all_triangles[all_triangles_len, 2] = all_vertices_len - 2
                    all_normals[all_triangles_len, 0] = 0.0
                    all_normals[all_triangles_len, 1] = 0.0
                    all_normals[all_triangles_len, 2] = -1.0
                    all_colors[all_triangles_len, :] = color_maps[segmentations[i, j, k]]
                    all_triangles_len += 1

                # z positive
                if k + 1 < dimz and voxels[i, j, k + 1] == 0 or k + 1 >= dimz:
                    # vertices
                    all_vertices[all_vertices_len, 0] = i
                    all_vertices[all_vertices_len, 1] = j
                    all_vertices[all_vertices_len, 2] = k + 1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i + 1
                    all_vertices[all_vertices_len, 1] = j
                    all_vertices[all_vertices_len, 2] = k + 1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i + 1
                    all_vertices[all_vertices_len, 1] = j + 1
                    all_vertices[all_vertices_len, 2] = k + 1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len, 0] = i
                    all_vertices[all_vertices_len, 1] = j + 1
                    all_vertices[all_vertices_len, 2] = k + 1
                    all_vertices_len += 1

                    # triangles
                    all_triangles[all_triangles_len, 0] = all_vertices_len - 4
                    all_triangles[all_triangles_len, 1] = all_vertices_len - 3
                    all_triangles[all_triangles_len, 2] = all_vertices_len - 2
                    all_normals[all_triangles_len, 0] = 0.0
                    all_normals[all_triangles_len, 1] = 0.0
                    all_normals[all_triangles_len, 2] = 1.0
                    all_colors[all_triangles_len, :] = color_maps[segmentations[i, j, k]]
                    all_triangles_len += 1
                    all_triangles[all_triangles_len, 0] = all_vertices_len - 4
                    all_triangles[all_triangles_len, 1] = all_vertices_len - 2
                    all_triangles[all_triangles_len, 2] = all_vertices_len - 1
                    all_normals[all_triangles_len, 0] = 0.0
                    all_normals[all_triangles_len, 1] = 0.0
                    all_normals[all_triangles_len, 2] = 1.0
                    all_colors[all_triangles_len, :] = color_maps[segmentations[i, j, k]]
                    all_triangles_len += 1

    return all_vertices[:all_vertices_len], all_triangles[:all_triangles_len], all_colors[:all_triangles_len], all_normals[:all_triangles_len]
