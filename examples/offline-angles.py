import trimesh
import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import SE3UncertaintyLib as SE3lib
import transformation as tr
import pickle

extents = [0.05,0.02,0.34]
woodstick = trimesh.creation.box(extents)

randR = tr.random_rotation_matrix()
ref_axis = np.dot(randR[:3,:3],np.array([1.,0.,0.]))

# angles = []
# for n in woodstick.face_normals:
#   angles.append(np.dot(n,ref_axis))

angles = []
mesh = []
for i in range(len(woodstick.vertices)):
  n = woodstick.face_normals[i]
  angle = np.dot(n,ref_axis)
  mesh.append([np.asarray(woodstick.vertices[i]),n,angle])
  print angle

test_axis = np.array([1.,0.,0.001])
test_angle = np.dot(test_axis,ref_axis)



# sorted_mesh = [sorted(mesh,key=lambda f: f[2]),ref_axis]
# pickle.dump(mesh_obj, open('sorted_mesh.p', 'wb'))


