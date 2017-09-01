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
import IPython

def normal_hashing(obj,num_random_unit,plot_histogram):
  entropy = 0.
  for k in range(num_random_unit):
    randR = tr.random_rotation_matrix()
    ref_axis = np.dot(randR[:3,:3],np.array([0.,0.,1.]))
    
    mesh = []
    angle_dict = []
    for i in range(len(obj.faces)):
      n = obj.face_normals[i]
      angle = np.arccos(np.dot(n,ref_axis))
      angle_dict.append(angle)
      # mesh.append([np.asarray(obj.faces[i]),n,angle])
      mesh.append([i,angle])

    hist,bin_edges = np.histogram(angle_dict,range=(0,np.pi),density=True)
    normalized_hist = hist/np.sum(hist)
    if sp.stats.entropy(normalized_hist) > entropy: #histogram with bigger shannon entropy is selected
      entropy = sp.stats.entropy(normalized_hist)
      # mesh_w_sorted_dict = [sorted(mesh,key=lambda f: f[2]),obj.vertices,ref_axis]
      sorted_dict = [sorted(mesh,key=lambda f: f[1]),ref_axis]
      toshow = normalized_hist,bin_edges

  print 'Selected unit vec:',sorted_dict[1]
  print 'Entropy:', entropy

  if plot_histogram:
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    width = 0.7*(toshow[1][1] - toshow[1][0])
    center = (toshow[1][:-1] + toshow[1][1:])/2.
    ax1.bar(center, toshow[0], align='center', width=width)
    ax1.set_title("Hist of selected unit vec")

    ax2 = fig.add_subplot(212)
    width = 0.7*(bin_edges[1] - bin_edges[0])
    center = (bin_edges[:-1] + bin_edges[1:])/2.
    ax2.bar(center, normalized_hist, align='center', width=width)
    ax2.set_title("Hist of a random unit vec")
    fig.tight_layout()
    plt.show(True)
  # IPython.embed()
  face_idx = [sorted_dict[0][i][0] for i in range(len(sorted_dict[0]))]
  angle_list = [sorted_dict[0][i][1] for i in range(len(sorted_dict[0]))]
  mesh_w_sorted_dict = [face_idx,angle_list,sorted_dict[1]]
  return mesh_w_sorted_dict # A list [[sorted_face_idx,angle],ref_axis]


extents = [0.05,0.02,0.34]
woodstick = trimesh.creation.box(extents)
complicated_obj = trimesh.load_mesh('featuretype.STL')
# mesh_w_dict = normal_hashing(complicated_obj,50,plot_histogram=True)
mesh_w_dict = normal_hashing(woodstick,10,plot_histogram=True)
pickle.dump(mesh_w_dict, open('woodstick_w_dict.p', 'wb'))
# raw_input()




# randR = tr.random_rotation_matrix()
# ref_axis = np.dot(randR[:3,:3],np.array([0.,0.,1.]))

# # angles = []
# # for n in woodstick.face_normals:
# #   angles.append(np.dot(n,ref_axis))

# # print woodstick.face_normals
# angles = []
# mesh = []
# for i in range(len(woodstick.vertices)):
#   n = woodstick.face_normals[i]
#   angle = np.arccos(np.dot(n,ref_axis))
#   mesh.append([np.asarray(woodstick.faces[i]),n,angle])
#   # print angle, n

# sorted_mesh = [sorted(mesh,key=lambda f: f[2]),woodstick.vertices,ref_axis]

# randR = tr.random_rotation_matrix()
# test_axis = np.array([0.,0.001,1.])
# test_axis = np.dot(randR[:3,:3],test_axis/np.linalg.norm(test_axis))
# test_angle = np.arccos(np.dot(test_axis,ref_axis))

# print 'test angle', test_angle, test_axis

# x_diff=[]
# for i in range(len(sorted_mesh[0])):
#   print sorted_mesh[0][i][1],sorted_mesh[0][i][2]
#   if i ==0:
#     x_diff.append(0.0)
#   else:
#     x_diff.append(sorted_mesh[0][i][2] - sorted_mesh[0][i-1][2])

# max_x_diff = np.max(x_diff)
# sorted_mesh.append(max_x_diff)

# x = [r[2] for r in sorted_mesh[0]]
# delta_angle = max_x_diff/2.
# up_bound = bisect.bisect_right(x,test_angle+delta_angle)
# low_bound = bisect.bisect_left(x[:up_bound],test_angle-delta_angle)

# print sorted_mesh[0][low_bound:up_bound]
# print 'max x diff', max_x_diff
# pickle.dump(sorted_mesh, open('sorted_mesh.p', 'wb'))
# print sorted_mesh
# #TODO: [test with all test normal axis]
# # [incase return no element??]
# # print test_angle








