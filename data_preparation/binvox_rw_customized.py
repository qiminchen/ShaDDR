import numpy as np
import struct

def read_header(fp):
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'): raise IOError('Not a binvox file')
    dims = [int(i) for i in fp.readline().strip().split(b' ')[1:]]
    fp.readline() #omit translate
    fp.readline() #omit scale
    fp.readline() #omit "data\n"
    return dims

#read voxels as 3D uint8 numpy array from a binvox file
#if fix_coords then i->x, j->y, k->z; otherwise i->x, j->z, k->y
def read_voxels(filename, fix_coords=True):
    fp = open(filename, 'rb')
    dims = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    fp.close()
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dims).astype(np.uint8)
    if fix_coords:
        data = np.ascontiguousarray(np.transpose(data, (0, 2, 1)))
    return data

def bwrite(fp,s):
    fp.write(s.encode())

def write_pair(fp,state,ctr):
    fp.write(struct.pack('B',state))
    fp.write(struct.pack('B',ctr))

#write voxels into a binvox file
#it uses precomputed run-length encoding results (state_ctr)
#because the original run-length encoding implemented in python in binvox_rw.py is too slow
#https://github.com/dimatura/binvox-rw-py/blob/public/binvox_rw.py
def write(filename, voxel_size, state_ctr):
    fp = open(filename, 'wb')
    bwrite(fp,'#binvox 1\n')
    bwrite(fp,'dim '+str(voxel_size[0])+' '+str(voxel_size[1])+' '+str(voxel_size[2])+'\n')
    bwrite(fp,'translate 0 0 0\nscale 1\ndata\n')
    c = 0
    while state_ctr[c,0]!=2:
        write_pair(fp, state_ctr[c,0], state_ctr[c,1])
        c += 1
    fp.close()
