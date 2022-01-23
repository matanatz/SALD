import h5py
import numpy as np
import os

datapath='/home/atzmonm/data/datasets/modelnet40_ply_hdf5_2048'
h5files = [x for x in os.listdir(datapath) if 'h5' == x.split('.')[-1] and 'train' in x]
shapes = []
for file in h5files:
    f = h5py.File(os.path.join(datapath, file), mode='r')
    shapes.append(f['data'][:])

shapes=np.concatenate(shapes,axis=0)
length = shapes.shape[0]
s_all = []
for i in range(length):
    s = '\"{0}"'.format(str(i))
    s = s.replace("'", '\"')
    s_all.append(s)

print (s_all)