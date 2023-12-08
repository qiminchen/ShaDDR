#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

# Cython specific imports
import numpy as np
cimport numpy as np
import cython
np.import_array()


#Note: the size of the input voxels should not exceed 32x32x32!!!
#Color is used to store voxel coordinates + which face
def colored_dual_contouring(char[:,:,::1] voxels):

    #arrays to store vertices, triangles, and triangle colors
    cdef int all_vertices_len = 0
    cdef int all_triangles_len = 0
    cdef int all_vertices_max = 524288
    cdef int all_triangles_max = 524288
    all_vertices_ = np.zeros([all_vertices_max,3], np.float32)
    all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
    all_normals_ = np.zeros([all_triangles_max,3], np.float32)
    all_colors_ = np.zeros([all_triangles_max,3], np.float32)
    cdef float[:,::1] all_vertices = all_vertices_
    cdef int[:,::1] all_triangles = all_triangles_
    cdef float[:,::1] all_normals = all_normals_
    cdef float[:,::1] all_colors = all_colors_

    cdef int dimx = voxels.shape[0]
    cdef int dimy = voxels.shape[1]
    cdef int dimz = voxels.shape[2]

    cdef int i,j,k

    for i in range(0,dimx):
        for j in range(0,dimy):
            for k in range(0,dimz):
                if voxels[i,j,k]==0: continue

                #x negative
                if i>0 and voxels[i-1,j,k]==0 or i<=0:

                    #vertices
                    all_vertices[all_vertices_len,0] = i
                    all_vertices[all_vertices_len,1] = j
                    all_vertices[all_vertices_len,2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i
                    all_vertices[all_vertices_len,1] = j+1
                    all_vertices[all_vertices_len,2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i
                    all_vertices[all_vertices_len,1] = j+1
                    all_vertices[all_vertices_len,2] = k+1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i
                    all_vertices[all_vertices_len,1] = j
                    all_vertices[all_vertices_len,2] = k+1
                    all_vertices_len += 1

                    #triangles
                    all_triangles[all_triangles_len,0] = all_vertices_len-4
                    all_triangles[all_triangles_len,1] = all_vertices_len-2
                    all_triangles[all_triangles_len,2] = all_vertices_len-3
                    all_colors[all_triangles_len,0] = i*8
                    all_colors[all_triangles_len,1] = j*8
                    all_colors[all_triangles_len,2] = k*8
                    all_normals[all_triangles_len,0] = -1.0
                    all_normals[all_triangles_len,1] = 0.0
                    all_normals[all_triangles_len,2] = 0.0
                    all_triangles_len += 1
                    all_triangles[all_triangles_len,0] = all_vertices_len-4
                    all_triangles[all_triangles_len,1] = all_vertices_len-1
                    all_triangles[all_triangles_len,2] = all_vertices_len-2
                    all_colors[all_triangles_len,0] = i*8
                    all_colors[all_triangles_len,1] = j*8
                    all_colors[all_triangles_len,2] = k*8
                    all_normals[all_triangles_len,0] = -1.0
                    all_normals[all_triangles_len,1] = 0.0
                    all_normals[all_triangles_len,2] = 0.0
                    all_triangles_len += 1

                #x positive
                if i+1<dimx and voxels[i+1,j,k]==0 or i+1>=dimx:

                    #vertices
                    all_vertices[all_vertices_len,0] = i+1
                    all_vertices[all_vertices_len,1] = j
                    all_vertices[all_vertices_len,2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i+1
                    all_vertices[all_vertices_len,1] = j+1
                    all_vertices[all_vertices_len,2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i+1
                    all_vertices[all_vertices_len,1] = j+1
                    all_vertices[all_vertices_len,2] = k+1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i+1
                    all_vertices[all_vertices_len,1] = j
                    all_vertices[all_vertices_len,2] = k+1
                    all_vertices_len += 1

                    #triangles
                    all_triangles[all_triangles_len,0] = all_vertices_len-4
                    all_triangles[all_triangles_len,1] = all_vertices_len-3
                    all_triangles[all_triangles_len,2] = all_vertices_len-2
                    all_colors[all_triangles_len,0] = i*8+4
                    all_colors[all_triangles_len,1] = j*8
                    all_colors[all_triangles_len,2] = k*8
                    all_normals[all_triangles_len,0] = 1.0
                    all_normals[all_triangles_len,1] = 0.0
                    all_normals[all_triangles_len,2] = 0.0
                    all_triangles_len += 1
                    all_triangles[all_triangles_len,0] = all_vertices_len-4
                    all_triangles[all_triangles_len,1] = all_vertices_len-2
                    all_triangles[all_triangles_len,2] = all_vertices_len-1
                    all_colors[all_triangles_len,0] = i*8+4
                    all_colors[all_triangles_len,1] = j*8
                    all_colors[all_triangles_len,2] = k*8
                    all_normals[all_triangles_len,0] = 1.0
                    all_normals[all_triangles_len,1] = 0.0
                    all_normals[all_triangles_len,2] = 0.0
                    all_triangles_len += 1

                #y negative
                if j>0 and voxels[i,j-1,k]==0 or j<=0:

                    #vertices
                    all_vertices[all_vertices_len,0] = i
                    all_vertices[all_vertices_len,1] = j
                    all_vertices[all_vertices_len,2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i+1
                    all_vertices[all_vertices_len,1] = j
                    all_vertices[all_vertices_len,2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i+1
                    all_vertices[all_vertices_len,1] = j
                    all_vertices[all_vertices_len,2] = k+1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i
                    all_vertices[all_vertices_len,1] = j
                    all_vertices[all_vertices_len,2] = k+1
                    all_vertices_len += 1

                    #triangles
                    all_triangles[all_triangles_len,0] = all_vertices_len-4
                    all_triangles[all_triangles_len,1] = all_vertices_len-3
                    all_triangles[all_triangles_len,2] = all_vertices_len-2
                    all_colors[all_triangles_len,0] = i*8
                    all_colors[all_triangles_len,1] = j*8+4
                    all_colors[all_triangles_len,2] = k*8
                    all_normals[all_triangles_len,0] = 0.0
                    all_normals[all_triangles_len,1] = -1.0
                    all_normals[all_triangles_len,2] = 0.0
                    all_triangles_len += 1
                    all_triangles[all_triangles_len,0] = all_vertices_len-4
                    all_triangles[all_triangles_len,1] = all_vertices_len-2
                    all_triangles[all_triangles_len,2] = all_vertices_len-1
                    all_colors[all_triangles_len,0] = i*8
                    all_colors[all_triangles_len,1] = j*8+4
                    all_colors[all_triangles_len,2] = k*8
                    all_normals[all_triangles_len,0] = 0.0
                    all_normals[all_triangles_len,1] = -1.0
                    all_normals[all_triangles_len,2] = 0.0
                    all_triangles_len += 1

                #y positive
                if j+1<dimy and voxels[i,j+1,k]==0 or j+1>=dimy:

                    #vertices
                    all_vertices[all_vertices_len,0] = i
                    all_vertices[all_vertices_len,1] = j+1
                    all_vertices[all_vertices_len,2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i+1
                    all_vertices[all_vertices_len,1] = j+1
                    all_vertices[all_vertices_len,2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i+1
                    all_vertices[all_vertices_len,1] = j+1
                    all_vertices[all_vertices_len,2] = k+1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i
                    all_vertices[all_vertices_len,1] = j+1
                    all_vertices[all_vertices_len,2] = k+1
                    all_vertices_len += 1

                    #triangles
                    all_triangles[all_triangles_len,0] = all_vertices_len-4
                    all_triangles[all_triangles_len,1] = all_vertices_len-2
                    all_triangles[all_triangles_len,2] = all_vertices_len-3
                    all_colors[all_triangles_len,0] = i*8+4
                    all_colors[all_triangles_len,1] = j*8+4
                    all_colors[all_triangles_len,2] = k*8
                    all_normals[all_triangles_len,0] = 0.0
                    all_normals[all_triangles_len,1] = 1.0
                    all_normals[all_triangles_len,2] = 0.0
                    all_triangles_len += 1
                    all_triangles[all_triangles_len,0] = all_vertices_len-4
                    all_triangles[all_triangles_len,1] = all_vertices_len-1
                    all_triangles[all_triangles_len,2] = all_vertices_len-2
                    all_colors[all_triangles_len,0] = i*8+4
                    all_colors[all_triangles_len,1] = j*8+4
                    all_colors[all_triangles_len,2] = k*8
                    all_normals[all_triangles_len,0] = 0.0
                    all_normals[all_triangles_len,1] = 1.0
                    all_normals[all_triangles_len,2] = 0.0
                    all_triangles_len += 1

                #z negative
                if k>0 and voxels[i,j,k-1]==0 or k<=0:

                    #vertices
                    all_vertices[all_vertices_len,0] = i
                    all_vertices[all_vertices_len,1] = j
                    all_vertices[all_vertices_len,2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i+1
                    all_vertices[all_vertices_len,1] = j
                    all_vertices[all_vertices_len,2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i+1
                    all_vertices[all_vertices_len,1] = j+1
                    all_vertices[all_vertices_len,2] = k
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i
                    all_vertices[all_vertices_len,1] = j+1
                    all_vertices[all_vertices_len,2] = k
                    all_vertices_len += 1

                    #triangles
                    all_triangles[all_triangles_len,0] = all_vertices_len-4
                    all_triangles[all_triangles_len,1] = all_vertices_len-2
                    all_triangles[all_triangles_len,2] = all_vertices_len-3
                    all_colors[all_triangles_len,0] = i*8
                    all_colors[all_triangles_len,1] = j*8
                    all_colors[all_triangles_len,2] = k*8+4
                    all_normals[all_triangles_len,0] = 0.0
                    all_normals[all_triangles_len,1] = 0.0
                    all_normals[all_triangles_len,2] = -1.0
                    all_triangles_len += 1
                    all_triangles[all_triangles_len,0] = all_vertices_len-4
                    all_triangles[all_triangles_len,1] = all_vertices_len-1
                    all_triangles[all_triangles_len,2] = all_vertices_len-2
                    all_colors[all_triangles_len,0] = i*8
                    all_colors[all_triangles_len,1] = j*8
                    all_colors[all_triangles_len,2] = k*8+4
                    all_normals[all_triangles_len,0] = 0.0
                    all_normals[all_triangles_len,1] = 0.0
                    all_normals[all_triangles_len,2] = -1.0
                    all_triangles_len += 1

                #z positive
                if k+1<dimz and voxels[i,j,k+1]==0 or k+1>=dimz:

                    #vertices
                    all_vertices[all_vertices_len,0] = i
                    all_vertices[all_vertices_len,1] = j
                    all_vertices[all_vertices_len,2] = k+1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i+1
                    all_vertices[all_vertices_len,1] = j
                    all_vertices[all_vertices_len,2] = k+1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i+1
                    all_vertices[all_vertices_len,1] = j+1
                    all_vertices[all_vertices_len,2] = k+1
                    all_vertices_len += 1
                    all_vertices[all_vertices_len,0] = i
                    all_vertices[all_vertices_len,1] = j+1
                    all_vertices[all_vertices_len,2] = k+1
                    all_vertices_len += 1

                    #triangles
                    all_triangles[all_triangles_len,0] = all_vertices_len-4
                    all_triangles[all_triangles_len,1] = all_vertices_len-3
                    all_triangles[all_triangles_len,2] = all_vertices_len-2
                    all_colors[all_triangles_len,0] = i*8+4
                    all_colors[all_triangles_len,1] = j*8
                    all_colors[all_triangles_len,2] = k*8+4
                    all_normals[all_triangles_len,0] = 0.0
                    all_normals[all_triangles_len,1] = 0.0
                    all_normals[all_triangles_len,2] = 1.0
                    all_triangles_len += 1
                    all_triangles[all_triangles_len,0] = all_vertices_len-4
                    all_triangles[all_triangles_len,1] = all_vertices_len-2
                    all_triangles[all_triangles_len,2] = all_vertices_len-1
                    all_colors[all_triangles_len,0] = i*8+4
                    all_colors[all_triangles_len,1] = j*8
                    all_colors[all_triangles_len,2] = k*8+4
                    all_normals[all_triangles_len,0] = 0.0
                    all_normals[all_triangles_len,1] = 0.0
                    all_normals[all_triangles_len,2] = 1.0
                    all_triangles_len += 1

    return all_vertices_[:all_vertices_len], all_triangles_[:all_triangles_len], all_normals_[:all_triangles_len], all_colors_[:all_triangles_len]

