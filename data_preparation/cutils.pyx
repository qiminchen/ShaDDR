#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np


cdef int render_depth_img_size = 5120


def get_state_ctr(char[:, :, ::1] img, int[:, ::1] state_ctr):
    cdef int state = 0
    cdef int ctr = 0
    cdef int p = 0
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    cdef int dimx,dimy,dimz

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]

    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if img[i,j,k]>0:
                    img[i,j,k] = 1
                if img[i,j,k]==state:
                    ctr += 1
                    if ctr==255:
                        state_ctr[p,0] = state
                        state_ctr[p,1] = ctr
                        p += 1
                        ctr = 0
                else:
                    if ctr>0:
                        state_ctr[p,0] = state
                        state_ctr[p,1] = ctr
                        p += 1
                    state = img[i,j,k]
                    ctr = 1

    if ctr > 0:
        state_ctr[p,0] = state
        state_ctr[p,1] = ctr
        p += 1

    state_ctr[p,0] = 2




def depth_fusion_XZY(char[:, :, ::1] img, int[:, :, ::1] rendering):
    cdef int dimx,dimy,dimz

    cdef int hdis = render_depth_img_size//2 #half depth image size
    
    cdef int c = 0
    cdef int u = 0
    cdef int v = 0
    cdef int d = 0
    
    cdef int outside_flag = 0
    
    cdef int x = 0
    cdef int y = 0
    cdef int z = 0
    
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    
    dimx = img.shape[0]
    dimz = img.shape[1]
    dimy = img.shape[2]
    
    #--model
    # 0 - X - front
    # 1 - Z - left
    # 2 - Y - up
    #--rendering [render_depth_img_size,render_depth_img_size,17]
    # 0 - top-down from Y
    # 1,2,3,4 - from X | Z
    # 5,6,7,8 - from X&Z
    # 9,10,11,12 - from X&Y | Z&Y
    # 13,14,15,16 - from X&Y&Z
    #read my figure for details (if you can find it)
    
    #get rendering
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                if img[x,z,y]>0:
                    #z-buffering
                    
                    c = 0
                    u = x + hdis
                    v = z + hdis
                    d = -y #y must always be negative in d to render from top
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 1
                    u = y + hdis
                    v = z + hdis
                    d = x
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 2
                    u = y + hdis
                    v = z + hdis
                    d = -x
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 3
                    u = x + hdis
                    v = y + hdis
                    d = z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 4
                    u = x + hdis
                    v = y + hdis
                    d = -z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 5
                    u = y + hdis
                    v = x-z + hdis
                    d = x+z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 6
                    u = y + hdis
                    v = x+z + hdis
                    d = x-z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 7
                    u = y + hdis
                    v = -x-z + hdis
                    d = -x+z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 8
                    u = y + hdis
                    v = -x+z + hdis
                    d = -x-z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 9
                    u = z + hdis
                    v = x+y + hdis
                    d = x-y
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 10
                    u = z + hdis
                    v = -x+y + hdis
                    d = -x-y
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 11
                    u = x + hdis
                    v = z+y + hdis
                    d = z-y
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 12
                    u = x + hdis
                    v = -z+y + hdis
                    d = -z-y
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u,v+1,c]>d: #block 2
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 2
                        rendering[u,v-1,c]=d
                        
                    c = 13
                    u = x+y + hdis
                    v = -y-z + hdis
                    d = x-y+z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u+1,v,c]>d: #block 6
                        rendering[u+1,v,c]=d
                    if rendering[u-1,v,c]>d: #block 6
                        rendering[u-1,v,c]=d
                    if rendering[u,v+1,c]>d: #block 6
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 6
                        rendering[u,v-1,c]=d
                    if rendering[u+1,v-1,c]>d: #block 6
                        rendering[u+1,v-1,c]=d
                    if rendering[u-1,v+1,c]>d: #block 6
                        rendering[u-1,v+1,c]=d
                        
                    c = 14
                    u = -x+y + hdis
                    v = -y-z + hdis
                    d = -x-y+z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u+1,v,c]>d: #block 6
                        rendering[u+1,v,c]=d
                    if rendering[u-1,v,c]>d: #block 6
                        rendering[u-1,v,c]=d
                    if rendering[u,v+1,c]>d: #block 6
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 6
                        rendering[u,v-1,c]=d
                    if rendering[u+1,v-1,c]>d: #block 6
                        rendering[u+1,v-1,c]=d
                    if rendering[u-1,v+1,c]>d: #block 6
                        rendering[u-1,v+1,c]=d
                        
                    c = 15
                    u = x+y + hdis
                    v = -y+z + hdis
                    d = x-y-z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u+1,v,c]>d: #block 6
                        rendering[u+1,v,c]=d
                    if rendering[u-1,v,c]>d: #block 6
                        rendering[u-1,v,c]=d
                    if rendering[u,v+1,c]>d: #block 6
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 6
                        rendering[u,v-1,c]=d
                    if rendering[u+1,v-1,c]>d: #block 6
                        rendering[u+1,v-1,c]=d
                    if rendering[u-1,v+1,c]>d: #block 6
                        rendering[u-1,v+1,c]=d
                        
                    c = 16
                    u = -x+y + hdis
                    v = -y+z + hdis
                    d = -x-y-z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                    if rendering[u+1,v,c]>d: #block 6
                        rendering[u+1,v,c]=d
                    if rendering[u-1,v,c]>d: #block 6
                        rendering[u-1,v,c]=d
                    if rendering[u,v+1,c]>d: #block 6
                        rendering[u,v+1,c]=d
                    if rendering[u,v-1,c]>d: #block 6
                        rendering[u,v-1,c]=d
                    if rendering[u+1,v-1,c]>d: #block 6
                        rendering[u+1,v-1,c]=d
                    if rendering[u-1,v+1,c]>d: #block 6
                        rendering[u-1,v+1,c]=d
                    
    
    
    #depth fusion
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                outside_flag = 0
                
                c = 0
                u = x + hdis
                v = z + hdis
                d = -y
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 1
                u = y + hdis
                v = z + hdis
                d = x
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 2
                u = y + hdis
                v = z + hdis
                d = -x
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 3
                u = x + hdis
                v = y + hdis
                d = z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 4
                u = x + hdis
                v = y + hdis
                d = -z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 5
                u = y + hdis
                v = x-z + hdis
                d = x+z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 6
                u = y + hdis
                v = x+z + hdis
                d = x-z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 7
                u = y + hdis
                v = -x-z + hdis
                d = -x+z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 8
                u = y + hdis
                v = -x+z + hdis
                d = -x-z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 9
                u = z + hdis
                v = x+y + hdis
                d = x-y
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 10
                u = z + hdis
                v = -x+y + hdis
                d = -x-y
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 11
                u = x + hdis
                v = z+y + hdis
                d = z-y
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 12
                u = x + hdis
                v = -z+y + hdis
                d = -z-y
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 13
                u = x+y + hdis
                v = -y-z + hdis
                d = x-y+z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 14
                u = -x+y + hdis
                v = -y-z + hdis
                d = -x-y+z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 15
                u = x+y + hdis
                v = -y+z + hdis
                d = x-y-z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 16
                u = -x+y + hdis
                v = -y+z + hdis
                d = -x-y-z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                
                if outside_flag==0:
                    img[x,z,y] = 1



def depth_fusion_XZY_5views(char[:, :, ::1] img, int[:, :, ::1] rendering):
    cdef int dimx,dimy,dimz
    
    cdef int hdis = render_depth_img_size//2 #half depth image size
    
    cdef int c = 0
    cdef int u = 0
    cdef int v = 0
    cdef int d = 0
    
    cdef int outside_flag = 0
    
    cdef int x = 0
    cdef int y = 0
    cdef int z = 0
    
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    
    dimx = img.shape[0]
    dimz = img.shape[1]
    dimy = img.shape[2]
    
    #--model
    # 0 - X - front
    # 1 - Z - left
    # 2 - Y - up
    #--rendering [render_depth_img_size,render_depth_img_size,17]
    # 0 - top-down from Y
    # 1,2,3,4 - from X | Z
    #read my figure for details (if you can find it)
    
    #get rendering
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                if img[x,z,y]>0:
                    #z-buffering
                    
                    c = 0
                    u = x + hdis
                    v = z + hdis
                    d = -y #y must always be negative in d to render from top
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 1
                    u = y + hdis
                    v = z + hdis
                    d = x
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 2
                    u = y + hdis
                    v = z + hdis
                    d = -x
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 3
                    u = x + hdis
                    v = y + hdis
                    d = z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
                        
                    c = 4
                    u = x + hdis
                    v = y + hdis
                    d = -z
                    if rendering[u,v,c]>d:
                        rendering[u,v,c]=d
  
    
    
    #depth fusion
    for x in range(dimx):
        for y in range(dimy):
            for z in range(dimz):
                outside_flag = 0
                
                c = 0
                u = x + hdis
                v = z + hdis
                d = -y
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 1
                u = y + hdis
                v = z + hdis
                d = x
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 2
                u = y + hdis
                v = z + hdis
                d = -x
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 3
                u = x + hdis
                v = y + hdis
                d = z
                if rendering[u,v,c]>d:
                    outside_flag += 1
                    
                c = 4
                u = x + hdis
                v = y + hdis
                d = -z
                if rendering[u,v,c]>d:
                    outside_flag += 1

                if outside_flag==0:
                    img[x,z,y] = 1




def floodfill(char[:, :, ::1] img, int[:, ::1] queue):
    cdef int dimx,dimy,dimz,max_queue_len,i,j,k
    cdef int pi = 0
    cdef int pj = 0
    cdef int pk = 0
    cdef int queue_start = 0
    cdef int queue_end = 1

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]
    max_queue_len = queue.shape[0]

    img[0,0,0] = 0
    queue[queue_start,0] = 0
    queue[queue_start,1] = 0
    queue[queue_start,2] = 0

    while queue_start != queue_end:
        pi = queue[queue_start,0]
        pj = queue[queue_start,1]
        pk = queue[queue_start,2]
        queue_start += 1
        if queue_start==max_queue_len:
            queue_start = 0

        pi = pi+1
        if pi<dimx and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pi = pi-2
        if pi>=0 and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pi = pi+1
        pj = pj+1
        if pj<dimy and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pj = pj-2
        if pj>=0 and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pj = pj+1
        pk = pk+1
        if pk<dimz and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

        pk = pk-2
        if pk>=0 and img[pi,pj,pk]==1:
            img[pi,pj,pk] = 0
            queue[queue_end,0] = pi
            queue[queue_end,1] = pj
            queue[queue_end,2] = pk
            queue_end += 1
            if queue_end==max_queue_len:
                queue_end = 0

    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if img[i,j,k]>0:
                    img[i,j,k] = 1




def cube_alpha_hull(char[:, :, ::1] img, int[:, :, ::1] accu, int cubesize_x, int cubesize_y, int cubesize_z):
    cdef int i,j,k
    cdef int dimx,dimy,dimz
    cdef int p, a000,a001,a010,a011,a100,a101,a110,a111
    cdef int cube_minx, cube_miny, cube_minz, cube_maxx, cube_maxy, cube_maxz

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]

    #first pass

    #dynamic programming to get integral map
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if img[i,j,k]==0:
                    a111 = 0
                else:
                    a111 = 1
                if i==0:
                    a011 = 0
                else:
                    a011 = accu[i-1,j,k]
                if j==0:
                    a101 = 0
                else:
                    a101 = accu[i,j-1,k]
                if k==0:
                    a110 = 0
                else:
                    a110 = accu[i,j,k-1]
                if j==0 or k==0:
                    a100 = 0
                else:
                    a100 = accu[i,j-1,k-1]
                if i==0 or k==0:
                    a010 = 0
                else:
                    a010 = accu[i-1,j,k-1]
                if i==0 or j==0:
                    a001 = 0
                else:
                    a001 = accu[i-1,j-1,k]
                if i==0 or j==0 or k==0:
                    a000 = 0
                else:
                    a000 = accu[i-1,j-1,k-1]
                accu[i,j,k] = a111 + a011 + a101 + a110 + a000 - a100 - a010 - a001
    
    #one
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                img[i,j,k] = 1
    
    #cube alpha hull
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                cube_minx = i-cubesize_x
                cube_miny = j-cubesize_y
                cube_minz = k-cubesize_z
                cube_maxx = i+cubesize_x-1
                cube_maxy = j+cubesize_y-1
                cube_maxz = k+cubesize_z-1
                if cube_maxx>=dimx: cube_maxx = dimx-1
                if cube_maxy>=dimy: cube_maxy = dimy-1
                if cube_maxz>=dimz: cube_maxz = dimz-1

                a111 = accu[cube_maxx,cube_maxy,cube_maxz]
                if cube_minx<0:
                    a011 = 0
                else:
                    a011 = accu[cube_minx,cube_maxy,cube_maxz]
                if cube_miny<0:
                    a101 = 0
                else:
                    a101 = accu[cube_maxx,cube_miny,cube_maxz]
                if cube_minz<0:
                    a110 = 0
                else:
                    a110 = accu[cube_maxx,cube_maxy,cube_minz]
                if cube_miny<0 or cube_minz<0:
                    a100 = 0
                else:
                    a100 = accu[cube_maxx,cube_miny,cube_minz]
                if cube_minx<0 or cube_minz<0:
                    a010 = 0
                else:
                    a010 = accu[cube_minx,cube_maxy,cube_minz]
                if cube_minx<0 or cube_miny<0:
                    a001 = 0
                else:
                    a001 = accu[cube_minx,cube_miny,cube_maxz]
                if cube_minx<0 or cube_miny<0 or cube_minz<0:
                    a000 = 0
                else:
                    a000 = accu[cube_minx,cube_miny,cube_minz]
                p = a111 + a100 + a010 + a001 - a011 - a101 - a110 - a000
                if p==0:
                    img[i,j,k] = 0

    #second pass

    #dynamic programming to get integral map
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if img[i,j,k]==0:
                    a111 = 0
                else:
                    a111 = 1
                if i==0:
                    a011 = 0
                else:
                    a011 = accu[i-1,j,k]
                if j==0:
                    a101 = 0
                else:
                    a101 = accu[i,j-1,k]
                if k==0:
                    a110 = 0
                else:
                    a110 = accu[i,j,k-1]
                if j==0 or k==0:
                    a100 = 0
                else:
                    a100 = accu[i,j-1,k-1]
                if i==0 or k==0:
                    a010 = 0
                else:
                    a010 = accu[i-1,j,k-1]
                if i==0 or j==0:
                    a001 = 0
                else:
                    a001 = accu[i-1,j-1,k]
                if i==0 or j==0 or k==0:
                    a000 = 0
                else:
                    a000 = accu[i-1,j-1,k-1]
                accu[i,j,k] = a111 + a011 + a101 + a110 + a000 - a100 - a010 - a001

    #cube alpha hull
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if img[i,j,k] != 0:
                    cube_minx = i-cubesize_x
                    cube_miny = j-cubesize_y
                    cube_minz = k-cubesize_z
                    cube_maxx = i+cubesize_x-1
                    cube_maxy = j+cubesize_y-1
                    cube_maxz = k+cubesize_z-1
                    if cube_maxx>=dimx: cube_maxx = dimx-1
                    if cube_maxy>=dimy: cube_maxy = dimy-1
                    if cube_maxz>=dimz: cube_maxz = dimz-1

                    a111 = accu[cube_maxx,cube_maxy,cube_maxz]
                    if cube_minx<0:
                        a011 = 0
                    else:
                        a011 = accu[cube_minx,cube_maxy,cube_maxz]
                    if cube_miny<0:
                        a101 = 0
                    else:
                        a101 = accu[cube_maxx,cube_miny,cube_maxz]
                    if cube_minz<0:
                        a110 = 0
                    else:
                        a110 = accu[cube_maxx,cube_maxy,cube_minz]
                    if cube_miny<0 or cube_minz<0:
                        a100 = 0
                    else:
                        a100 = accu[cube_maxx,cube_miny,cube_minz]
                    if cube_minx<0 or cube_minz<0:
                        a010 = 0
                    else:
                        a010 = accu[cube_minx,cube_maxy,cube_minz]
                    if cube_minx<0 or cube_miny<0:
                        a001 = 0
                    else:
                        a001 = accu[cube_minx,cube_miny,cube_maxz]
                    if cube_minx<0 or cube_miny<0 or cube_minz<0:
                        a000 = 0
                    else:
                        a000 = accu[cube_minx,cube_miny,cube_minz]
                    p = a111 + a100 + a010 + a001 - a011 - a101 - a110 - a000

                    if cube_minx<-1: cube_minx = -1
                    if cube_miny<-1: cube_miny = -1
                    if cube_minz<-1: cube_minz = -1
                    if p != (cube_maxx-cube_minx)*(cube_maxy-cube_miny)*(cube_maxz-cube_minz):
                        img[i,j,k] = 0



#numpy's transpose-assign is too slow
def get_transpose(char[:, :, ::1] tmp_voxel, char[:, :, ::1] batch_voxels, int padding, int target_axis, int flip):
    cdef int dim,dim1, x,y,z

    dim = batch_voxels.shape[0]
    dim1 = dim-1

    if target_axis==0 and flip==0:
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    tmp_voxel[x+padding,y+padding,z+padding] = batch_voxels[x,y,z]

    if target_axis==0 and flip==1:
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    tmp_voxel[x+padding,y+padding,z+padding] = batch_voxels[dim1-x,y,z]

    if target_axis==1 and flip==0:
        for y in range(dim):
            for x in range(dim):
                for z in range(dim):
                    tmp_voxel[x+padding,y+padding,z+padding] = batch_voxels[y,x,z]

    if target_axis==1 and flip==1:
        for y in range(dim):
            for x in range(dim):
                for z in range(dim):
                    tmp_voxel[x+padding,y+padding,z+padding] = batch_voxels[y,dim1-x,z]

    if target_axis==2 and flip==0:
        for z in range(dim):
            for y in range(dim):
                for x in range(dim):
                    tmp_voxel[x+padding,y+padding,z+padding] = batch_voxels[z,y,x]

    if target_axis==2 and flip==1:
        for z in range(dim):
            for y in range(dim):
                for x in range(dim):
                    tmp_voxel[x+padding,y+padding,z+padding] = batch_voxels[z,y,dim1-x]


#numpy's transpose-assign is too slow
def recover_transpose(char[:, :, ::1] tmp_voxel, char[:, :, ::1] batch_voxels, int padding, int target_axis, int flip):
    cdef int dim,dim1, x,y,z

    dim = batch_voxels.shape[0]
    dim1 = dim-1

    if target_axis==0 and flip==0:
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    batch_voxels[x,y,z] = tmp_voxel[x+padding,y+padding,z+padding]

    if target_axis==0 and flip==1:
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    batch_voxels[dim1-x,y,z] = tmp_voxel[x+padding,y+padding,z+padding]

    if target_axis==1 and flip==0:
        for y in range(dim):
            for x in range(dim):
                for z in range(dim):
                    batch_voxels[y,x,z] = tmp_voxel[x+padding,y+padding,z+padding]

    if target_axis==1 and flip==1:
        for y in range(dim):
            for x in range(dim):
                for z in range(dim):
                    batch_voxels[y,dim1-x,z] = tmp_voxel[x+padding,y+padding,z+padding]

    if target_axis==2 and flip==0:
        for z in range(dim):
            for y in range(dim):
                for x in range(dim):
                    batch_voxels[z,y,x] = tmp_voxel[x+padding,y+padding,z+padding]

    if target_axis==2 and flip==1:
        for z in range(dim):
            for y in range(dim):
                for x in range(dim):
                    batch_voxels[z,y,dim1-x] = tmp_voxel[x+padding,y+padding,z+padding]


#assume target direction is X-
def boundary_cull(char[:, :, ::1] img, int[:, :, ::1] accu, char[:, :, ::1] refimg, int[:, :, ::1] refaccu, int[:, ::1] queue):
    cdef int i,j,k,x,y,z,q_start,q_end,this_depth,tmp_depth
    cdef int dimx,dimy,dimz,queue_len,depth_channel

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]
    depth_channel = dimx-1
    queue_len = queue.shape[0]

    #get accu
    for y in range(dimy):
        for z in range(dimz):
            accu[0,y,z] = 0
            accu[depth_channel,y,z] = 0
    for x in range(1,depth_channel):
        for y in range(dimy):
            for z in range(dimz):
                if img[x,y,z]>0:
                    accu[x,y,z] = accu[x-1,y,z] + 1
                    accu[depth_channel,y,z] = x
                else:
                    accu[x,y,z] = accu[x-1,y,z]

    #get refaccu
    for y in range(dimy):
        for z in range(dimz):
            refaccu[0,y,z] = 0
    for x in range(1,depth_channel):
        for y in range(dimy):
            for z in range(dimz):
                if refimg[x,y,z]>0:
                    refaccu[x,y,z] = refaccu[x-1,y,z] + 1
                else:
                    refaccu[x,y,z] = refaccu[x-1,y,z]
    
    #find boundary voxels and put into queue
    q_start = 0
    q_end = 0
    for y in range(1,dimy-1):
        for z in range(1,dimz-1):
            queue[q_end,0] = y
            queue[q_end,1] = z
            q_end += 1
            if q_end==queue_len: q_end = 0
    
    while q_start!=q_end:
        y = queue[q_start,0]
        z = queue[q_start,1]
        q_start += 1
        if q_start==queue_len: q_start = 0


        this_depth = accu[depth_channel,y,z]
        if refimg[this_depth,y,z]==0:
            tmp_depth = accu[depth_channel,y-1,z]
            if this_depth>tmp_depth:
                if this_depth-tmp_depth != accu[this_depth,y,z] - accu[tmp_depth,y,z]:
                    x = this_depth
                    while img[x,y,z]>0:
                        x -= 1
                    if refaccu[this_depth,y,z] - refaccu[x,y,z] == 0:

                        #remove voxels
                        x = this_depth
                        while img[x,y,z]>0:
                            img[x,y,z] = 0
                            x -= 1

                        #update accu
                        accu[depth_channel,y,z] = 0
                        for x in range(1,depth_channel):
                            if img[x,y,z]>0:
                                accu[x,y,z] = accu[x-1,y,z] + 1
                                accu[depth_channel,y,z] = x
                            else:
                                accu[x,y,z] = accu[x-1,y,z]

                        #put neighbors into queue
                        queue[q_end,0] = y
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y-1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y+1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z-1
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z+1
                        q_end += 1
                        if q_end==queue_len: q_end = 0


        this_depth = accu[depth_channel,y,z]
        if refimg[this_depth,y,z]==0:
            tmp_depth = accu[depth_channel,y+1,z]
            if this_depth>tmp_depth:
                if this_depth-tmp_depth != accu[this_depth,y,z] - accu[tmp_depth,y,z]:
                    x = this_depth
                    while img[x,y,z]>0:
                        x -= 1
                    if refaccu[this_depth,y,z] - refaccu[x,y,z] == 0:

                        #remove voxels
                        x = this_depth
                        while img[x,y,z]>0:
                            img[x,y,z] = 0
                            x -= 1

                        #update accu
                        accu[depth_channel,y,z] = 0
                        for x in range(1,depth_channel):
                            if img[x,y,z]>0:
                                accu[x,y,z] = accu[x-1,y,z] + 1
                                accu[depth_channel,y,z] = x
                            else:
                                accu[x,y,z] = accu[x-1,y,z]

                        #put neighbors into queue
                        queue[q_end,0] = y
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y-1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y+1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z-1
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z+1
                        q_end += 1
                        if q_end==queue_len: q_end = 0

        this_depth = accu[depth_channel,y,z]
        if refimg[this_depth,y,z]==0:
            tmp_depth = accu[depth_channel,y,z-1]
            if this_depth>tmp_depth:
                if this_depth-tmp_depth != accu[this_depth,y,z] - accu[tmp_depth,y,z]:
                    x = this_depth
                    while img[x,y,z]>0:
                        x -= 1
                    if refaccu[this_depth,y,z] - refaccu[x,y,z] == 0:

                        #remove voxels
                        x = this_depth
                        while img[x,y,z]>0:
                            img[x,y,z] = 0
                            x -= 1

                        #update accu
                        accu[depth_channel,y,z] = 0
                        for x in range(1,depth_channel):
                            if img[x,y,z]>0:
                                accu[x,y,z] = accu[x-1,y,z] + 1
                                accu[depth_channel,y,z] = x
                            else:
                                accu[x,y,z] = accu[x-1,y,z]

                        #put neighbors into queue
                        queue[q_end,0] = y
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y-1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y+1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z-1
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z+1
                        q_end += 1
                        if q_end==queue_len: q_end = 0

        this_depth = accu[depth_channel,y,z]
        if refimg[this_depth,y,z]==0:
            tmp_depth = accu[depth_channel,y,z+1]
            if this_depth>tmp_depth:
                if this_depth-tmp_depth != accu[this_depth,y,z] - accu[tmp_depth,y,z]:
                    x = this_depth
                    while img[x,y,z]>0:
                        x -= 1
                    if refaccu[this_depth,y,z] - refaccu[x,y,z] == 0:

                        #remove voxels
                        x = this_depth
                        while img[x,y,z]>0:
                            img[x,y,z] = 0
                            x -= 1

                        #update accu
                        accu[depth_channel,y,z] = 0
                        for x in range(1,depth_channel):
                            if img[x,y,z]>0:
                                accu[x,y,z] = accu[x-1,y,z] + 1
                                accu[depth_channel,y,z] = x
                            else:
                                accu[x,y,z] = accu[x-1,y,z]

                        #put neighbors into queue
                        queue[q_end,0] = y
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y-1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y+1
                        queue[q_end,1] = z
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z-1
                        q_end += 1
                        if q_end==queue_len: q_end = 0
                        queue[q_end,0] = y
                        queue[q_end,1] = z+1
                        q_end += 1
                        if q_end==queue_len: q_end = 0

        #lower corners
        this_depth = accu[depth_channel,y,z]
        if refimg[this_depth,y,z]==0:
            k = 0
            if k==0:
                i = accu[depth_channel,y-1,z]
                j = accu[depth_channel,y,z-1]
                if i<this_depth and j<this_depth:
                    if i>j: tmp_depth = i
                    else: tmp_depth = j
                    k = 1
            if k==0:
                i = accu[depth_channel,y+1,z]
                j = accu[depth_channel,y,z-1]
                if i<this_depth and j<this_depth:
                    if i>j: tmp_depth = i
                    else: tmp_depth = j
                    k = 1
            if k==0:
                i = accu[depth_channel,y-1,z]
                j = accu[depth_channel,y,z+1]
                if i<this_depth and j<this_depth:
                    if i>j: tmp_depth = i
                    else: tmp_depth = j
                    k = 1
            if k==0:
                i = accu[depth_channel,y+1,z]
                j = accu[depth_channel,y,z+1]
                if i<this_depth and j<this_depth:
                    if i>j: tmp_depth = i
                    else: tmp_depth = j
                    k = 1

            if k>0:
                x = this_depth
                while img[x,y,z]>0 and refimg[x,y,z]==0 and x>tmp_depth:
                    img[x,y,z] = 0
                    x -= 1

                #update accu
                accu[depth_channel,y,z] = 0
                for x in range(1,depth_channel):
                    if img[x,y,z]>0:
                        accu[x,y,z] = accu[x-1,y,z] + 1
                        accu[depth_channel,y,z] = x
                    else:
                        accu[x,y,z] = accu[x-1,y,z]

                #put neighbors into queue
                queue[q_end,0] = y
                queue[q_end,1] = z
                q_end += 1
                if q_end==queue_len: q_end = 0
                queue[q_end,0] = y-1
                queue[q_end,1] = z
                q_end += 1
                if q_end==queue_len: q_end = 0
                queue[q_end,0] = y+1
                queue[q_end,1] = z
                q_end += 1
                if q_end==queue_len: q_end = 0
                queue[q_end,0] = y
                queue[q_end,1] = z-1
                q_end += 1
                if q_end==queue_len: q_end = 0
                queue[q_end,0] = y
                queue[q_end,1] = z+1
                q_end += 1
                if q_end==queue_len: q_end = 0



#record occluded voxel count
def get_rays(char[:, :, ::1] img, int[::1] ray_x1, int[::1] ray_y1, int[::1] ray_z1, int[::1] ray_x2, int[::1] ray_y2, int[::1] ray_z2, char[:, :, ::1] visibility_flag):
    cdef int dimx,dimy,dimz,dimz2
    cdef int i,j,k,p

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]
    dimz2 = dimz//2

    
    #get visibility_flag
    for i in range(dimx):
        for k in range(dimz):
            j = dimy-1
            if img[i,j,k]==0:
                visibility_flag[i,j,k] = 0
            for j in range(dimy-2,-1,-1):
                if img[i,j,k]==0:
                    if i==0 or i==dimx-1 or k==0 or k==dimz-1:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i-1,j+1,k]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i+1,j+1,k]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i,j+1,k-1]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i,j+1,k+1]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i-1,j+1,k-1]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i-1,j+1,k+1]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i+1,j+1,k-1]==0:
                        visibility_flag[i,j,k] = 0
                    elif visibility_flag[i+1,j+1,k+1]==0:
                        visibility_flag[i,j,k] = 0



    #y axis, top down

    for i in range(dimx):
        for k in range(dimz):
            p = 2
            for j in range(dimy):
                if img[i,j,k]>0:
                    if p==0:
                        p = 1
                        ray_y1[j] += 1
                    if p==2:
                        p = 1
                else:
                    if p==1:
                        p = 0
            p = 2
            for j in range(dimy-1,-1,-1):
                if img[i,j,k]>0:
                    if p==0:
                        p = 1
                        ray_y2[j] += 1
                    if p==2:
                        p = 1
                else:
                    if p==1:
                        p = 0

    #x axis, front back

    for j in range(dimy):
        for k in range(dimz):
            p = 2
            for i in range(dimx):
                if img[i,j,k]>0:
                    if p==0:
                        p = 1
                        if visibility_flag[i-1,j,k]==0:
                            ray_x1[i] += 1
                    if p==2:
                        p = 1
                else:
                    if p==1:
                        p = 0
            p = 2
            for i in range(dimx-1,-1,-1):
                if img[i,j,k]>0:
                    if p==0:
                        p = 1
                        if visibility_flag[i+1,j,k]==0:
                            ray_x2[i] += 1
                    if p==2:
                        p = 1
                else:
                    if p==1:
                        p = 0


    #special treatment for z axis, the symmetry one

    for i in range(dimx):
        for j in range(dimy):
            p = 2
            for k in range(dimz):
                if img[i,j,k]>0:
                    if p==0:
                        p = 1
                        if visibility_flag[i,j,k-1]==0:
                            ray_z1[k] += 1
                    if p==2:
                        p = 1
                else:
                    if p==1:
                        p = 0
                if k==dimz2:
                    if p==0:
                        p = 2
            p = 2
            for k in range(dimz-1,-1,-1):
                if img[i,j,k]>0:
                    if p==0:
                        p = 1
                        if visibility_flag[i,j,k+1]==0:
                            ray_z2[k] += 1
                    if p==2:
                        p = 1
                else:
                    if p==1:
                        p = 0
                if k==dimz2:
                    if p==0:
                        p = 2



#get approximate normal direction
#0 = x+ direction
#1 = x- direction
#2 = y+ direction
#3 = y- direction
#4 = z+ direction
#5 = z- direction
def get_voxel_approximate_normal_direction(char[:, :, ::1] img, char[:, :, ::1] norm):
    cdef int dimx,dimy,dimz
    cdef int i,j,k,x,y,z,x_count,y_count,z_count,x_sign,y_sign,z_sign

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]

    for i in range(dimx-2):
        for j in range(dimy-2):
            for k in range(dimz-2):
                if img[i+1,j+1,k+1]>0:
                    x_count = 0
                    y_count = 0
                    z_count = 0
                    x_sign = 0
                    y_sign = 0
                    z_sign = 0
                    for x in range(3):
                        for y in range(3):
                            z_count = z_count+img[i+x,j+y,k]
                            z_count = z_count-img[i+x,j+y,k+2]
                    for x in range(3):
                        for z in range(3):
                            y_count = y_count+img[i+x,j,k+z]
                            y_count = y_count-img[i+x,j+2,k+z]
                    for y in range(3):
                        for z in range(3):
                            x_count = x_count+img[i,j+y,k+z]
                            x_count = x_count-img[i+2,j+y,k+z]
                    if x_count<0:
                        x_count = -x_count
                        x_sign = 1
                    if y_count<0:
                        y_count = -y_count
                        y_sign = 1
                    if z_count<0:
                        z_count = -z_count
                        z_sign = 1
                    if x_count>=y_count and x_count>=z_count:
                        norm[i+1,j+1,k+1] = x_sign
                    elif y_count>=z_count:
                        norm[i+1,j+1,k+1] = 2+y_sign
                    else:
                        norm[i+1,j+1,k+1] = 4+z_sign



#mark surface voxels as 1 and others 0
def get_voxel_surface_flag(char[:, :, ::1] img, char[:, :, ::1] surf):
    cdef int dimx,dimy,dimz
    cdef int i,j,k,x,y,z,count

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]

    for i in range(dimx-2):
        for j in range(dimy-2):
            for k in range(dimz-2):
                if img[i+1,j+1,k+1]>0:
                    count = 0
                    for x in range(3):
                        for y in range(3):
                            for z in range(3):
                                count += img[i+x,j+y,k+z]
                    if count!=27:
                        surf[i+1,j+1,k+1] = 1



#inpaint colors of all surface voxels
def inpaint_surface(char[:, :, :, ::1] img, char[:, :, ::1] surf, char[:, :, ::1] filled):
    cdef int dimx,dimy,dimz
    cdef int i,j,k, x,y,z, stop, complete

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]

    stop = 0
    while not stop:
        stop = 1

        for i in range(dimx):
            for j in range(dimy):
                for k in range(dimz):
                    if filled[i,j,k]==1:
                        filled[i,j,k] = 2

        for i in range(1,dimx-1):
            for j in range(1,dimy-1):
                for k in range(1,dimz-1):
                    if filled[i,j,k]==2:
                        complete = 1
                        for x in range(-1,2):
                            for y in range(-1,2):
                                for z in range(-1,2):
                                    if surf[i+x,j+y,k+z] and not filled[i+x,j+y,k+z]:
                                        stop = 0
                                        complete = 0
                                        img[i+x,j+y,k+z,0] = img[i,j,k,0]
                                        img[i+x,j+y,k+z,1] = img[i,j,k,1]
                                        img[i+x,j+y,k+z,2] = img[i,j,k,2]
                                        filled[i+x,j+y,k+z] = 1
                        if complete:
                            filled[i,j,k] = 3



#inpaint each img[x,y,:] independently
def inpaint_volume_z(char[:, :, :, ::1] img, char[:, :, ::1] surf, char[:, :, ::1] filled, int[::1] dist):
    cdef int dimx,dimy,dimz
    cdef int i,j,k, x,y,z, r,g,b,d

    dimx = img.shape[0]
    dimy = img.shape[1]
    dimz = img.shape[2]


    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                dist[k] = 16384
            r = 0
            b = 0
            g = 0

            d = 16384
            for k in range(dimz):
                if surf[i,j,k]:
                    r = img[i,j,k,0]
                    g = img[i,j,k,1]
                    b = img[i,j,k,2]
                    d = 0
                elif d<dist[k]:
                    img[i,j,k,0] = r
                    img[i,j,k,1] = g
                    img[i,j,k,2] = b
                    filled[i,j,k] = 1
                    dist[k] = d
                d += 1

            d = 16384
            for k in range(dimz-1,-1,-1):
                if surf[i,j,k]:
                    r = img[i,j,k,0]
                    g = img[i,j,k,1]
                    b = img[i,j,k,2]
                    d = 0
                elif d<dist[k]:
                    img[i,j,k,0] = r
                    img[i,j,k,1] = g
                    img[i,j,k,2] = b
                    filled[i,j,k] = 1
                    dist[k] = d
                d += 1


