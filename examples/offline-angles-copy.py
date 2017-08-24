import trimesh
import numpy as np
import scipy as sp
import bisect
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import SE3UncertaintyLib as SE3lib
import transformation as tr
import pickle

obj = trimesh.load_mesh('featuretype.STL')

randR = tr.random_rotation_matrix()
ref_axis = np.dot(randR[:3,:3],np.array([0.,0.,1.]))

# angles = []
# for n in obj.face_normals:
#   angles.append(np.dot(n,ref_axis))

# print obj.face_normals
angles = []
mesh = []
for i in range(len(obj.vertices)):
  n = obj.face_normals[i]
  angle = np.arccos(np.dot(n,ref_axis))
  mesh.append([np.asarray(obj.faces[i]),n,angle])
  # print angle, n

sorted_mesh = [sorted(mesh,key=lambda f: f[2]),obj.vertices,ref_axis]

randR = tr.random_rotation_matrix()
test_axis = np.array([0.,0.001,1.])
test_axis = np.dot(randR[:3,:3],test_axis/np.linalg.norm(test_axis))
test_angle = np.arccos(np.dot(test_axis,ref_axis))

# print 'test angle', test_angle, test_axis

x_diff=[]
for i in range(len(sorted_mesh[0])):
  # print sorted_mesh[0][i][1],sorted_mesh[0][i][2]
  if i ==0:
    x_diff.append(0.0)
  else:
    x_diff.append(sorted_mesh[0][i][2] - sorted_mesh[0][i-1][2])

max_x_diff = np.max(x_diff)
sorted_mesh.append(max_x_diff)

x = [r[2] for r in sorted_mesh[0]]
delta_angle = max_x_diff/2.
up_bound = bisect.bisect_right(x,test_angle+delta_angle)
low_bound = bisect.bisect_left(x[:up_bound],test_angle-delta_angle)

print len(sorted_mesh[0][low_bound:up_bound])
print len(obj.faces)
print 'max x diff', max_x_diff
# pickle.dump(sorted_mesh, open('sorted_mesh_featuretype.p', 'wb'))
# print sorted_mesh
#TODO: [test with all test normal axis]
# [incase return no element??]
# print test_angle








