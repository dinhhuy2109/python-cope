import numpy as np
# from openravepy import *
import ParticleLibOutlier as ptcl
import trimesh
import transformation as tr
import SE3UncertaintyLib as SE3
import copy
import IPython
# env = Environment()
# env.SetViewer('qtcoin')
# woodstick = env.ReadKinBodyXMLFile("woodstick.xml")
# #env.AddKinBody(woodstick,True)
# ion()

# Measurements
o_p = 2e-3
o_n = 5./180.0*np.pi

p1 = [-0.025,0.007,0.11] # side near y
n1 = [1.,0,0.00001]
d1 = [p1,n1,o_p,o_n]

p3 = [0.02505,-0.003,0.12] # side near y
n3 = [-1.,0.00002,0]
d3 = [p3,n3,o_p,o_n]

p2 = [0.01,0.005, 0.17007] # top
n2 = [0,0.00002,-1.]
d2 = [p2,n2,o_p,o_n]

p4 = [-0.015,-0.025, 0.19] # top!!!!!!!!!!!!!!!
n4 = [0,0.0,-1.]
d4 = [p4,n4,o_p,o_n]

p5 = [0.002,-0.0101,0.102] # side near x
n5 = [0.,1.,0.0]
d5 = [p5,n5,o_p,o_n]

p6 = [0.01,0.0102,0.12] # side near x
n6 = [0.00001,-1,0.0]
d6 = [p6,n6,o_p,o_n]

p7 = [-0.01,-0.0102,0.02] # side near x
n7 = np.dot(n6,-1)
d7 = [p7,n7,o_p,o_n]

p8 = [0.005,-0.0102,-0.052] # side near x
d8 = [p8,n7,o_p,o_n]

p9 = [-0.008,0.0102,-0.1] # side near x
d9 = [p9,n6,o_p,o_n]

D = [d3,d2,d1,d5,d4,d6,d7,d8,d9]

extents = [0.05,0.02,0.34]
woodstick = trimesh.creation.box(extents)
# handles = []
# for d in D:
    # handles.append(env.plot3(d[0],0.001, colors=[0, 1, 0],drawstyle=1))
    
# # raw_input("Press Enter to continue...")
# tiny = 1e-5
sigma0 = np.diag([0.0009, 0.0009,0.0009,0.009,0.009,0.009],0) #trans,rot
cholsigma = np.linalg.cholesky(sigma0).T
uniformsample = np.random.uniform(-1,1,size = 6)
xi_new_particle = np.dot(cholsigma, uniformsample)
T = SE3.VecToTran(xi_new_particle)
# T = np.eye(4)
for d in D:
    d[0] = np.dot(T[:3,:3],d[0]) + T[:3,3]
    d[1] = np.dot(T[:3,:3],d[1])
sigma_desired = 0.25*np.diag([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6],0)
# print sigma0
dim = 6 # 6 DOFs
prune_percentage = 0.8
ptcls0 = [np.eye(4)]
M = 10# No. of particles per delta-neighbohood

# color = trimesh.visual.random_color()
# for face in woodstick.faces:
#     woodstick.visual.face_colors[face] = color

# show = woodstick.copy()
# show.apply_transform(T)
# color = trimesh.visual.random_color()
# for d in D:
#   sphere = trimesh.creation.icosphere(3,0.0025)
#   TF = np.eye(4)
#   TF[:3,3] = d[0]
#   TF2 = np.eye(4)
#   angle = np.arccos(np.dot(d[1],np.array([0,0,1])))
#   vec = np.cross(d[1],np.array([0,0,1]))
#   TF2[:3,:3] = SE3.VecToRot(angle*vec)
#   TF2[:3,3] = d[0] + np.dot(SE3.VecToRot(angle*vec),np.array([0,0,0.1/2.]))
#   cyl = trimesh.creation.cylinder(0.001,0.1)
#   cyl.apply_transform(TF2)
#   show += cyl
#   sphere.apply_transform(TF)
#   show+=sphere
# show.show()
# raw_input()



inliers_indexes = [5, 8, 6, 2, 3, 0, 1,7]
D1 = [D[i] for i in inliers_indexes]

list_particles, weights = ptcl.ScalingSeries(woodstick,ptcls0, D1, M, sigma0, sigma_desired,prune_percentage, dim,visualize = False)
maxweight = weights[0]
for w in weights:
  if w > maxweight:
    maxweight = w   
acum_weight = 0
acum_vec = np.zeros(6)
weight_threshold = 0.7*maxweight
for i in range(len(list_particles)):
  if weights[i] > weight_threshold:
    p = SE3.TranToVec(list_particles[i])
    acum_vec += p*weights[i]
    acum_weight += weights[i]
estimated_particle = acum_vec*(1./acum_weight)
transf = SE3.VecToTran(estimated_particle)
print "Resulting estimation:\n", transf
print "Real transformation\n", T

# D2 = [D[i] for i in inliers_indexes]
# sum_energy = 0.
# for k in range(len(D2)):
#   datapoint = copy.deepcopy(D2[k])
#   T_inv = np.linalg.inv(SE3.VecToTran(estimated_particle))
#   datapoint[0] = np.dot(T_inv[:3,:3],datapoint[0]) + T_inv[:3,3]
#   datapoint[1] = np.dot(T_inv[:3,:3],datapoint[1])
#   dist = ptcl.CalculateMahaDistanceMesh(woodstick,datapoint)
#   print 'Dist of ',inliers_indexes[k], 'is', dist
#   sum_energy += dist**2
# print sum_energy
# IPython.embed()
raw_input()




n = 5  #  the minimum number of data values required to fit the model
k = 10 # the maximum number of iterations allowed in the algorithm
t = 2  # a threshold value for determining when a data point fits a model
d = 7  # the number of close data values required to assert that a model fits well to data

iterations = 0
bestfit = np.eye(4)
besterr = 999.
from random import randrange
while iterations < k:
  iterations += 1
  maybeinliers = []
  maybeinliers_indexes = []
  while len(maybeinliers_indexes) < n:
    random_index = randrange(0,len(D))
    if random_index not in maybeinliers_indexes:
      maybeinliers.append(D[random_index])
      maybeinliers_indexes.append(random_index)
  data = maybeinliers
  alsoinliers = []

  # print 'init maybeinliers_indexes', maybeinliers_indexes

  list_particles, weights = ptcl.ScalingSeries(woodstick,ptcls0, data, M, sigma0, sigma_desired,prune_percentage, dim,visualize =False)
  maxweight = weights[0]
  for w in weights:
    if w > maxweight:
      maxweight = w   

  acum_weight = 0
  acum_vec = np.zeros(6)
  weight_threshold = 0.7*maxweight
  for i in range(len(list_particles)):
    if weights[i] > weight_threshold:
      p = SE3.TranToVec(list_particles[i])
      acum_vec += p*weights[i]
      acum_weight += weights[i]
  estimated_particle = acum_vec*(1./acum_weight)
  transf = SE3.VecToTran(estimated_particle)

  maybemodel = estimated_particle
  for i in range(len(D)):
    if i not in maybeinliers_indexes:
      #check if D[i] is fit the new particles?
      #if yes, add to the alsoinliers
      datapoint = copy.deepcopy(D[i])
      T_inv = np.linalg.inv(SE3.VecToTran(maybemodel))
      datapoint[0] = np.dot(T_inv[:3,:3],datapoint[0]) + T_inv[:3,3]
      datapoint[1] = np.dot(T_inv[:3,:3],datapoint[1])
      dist = ptcl.CalculateMahaDistanceMesh(woodstick,datapoint)
      # print 'MahaDist', dist
      if dist < t:
        maybeinliers.append(datapoint)
        maybeinliers_indexes.append(i)
        # print 'idx',i
        # print 'Mahadist', dist
        # raw_input('enter to contd')
  if len(maybeinliers) > d: #maybe good model
    # print "Maybe good model!"
    # raw_input()
    data = copy.deepcopy(maybeinliers)
    list_particles, weights = ptcl.ScalingSeries(woodstick,ptcls0, data, M, sigma0, sigma_desired,prune_percentage, dim,visualize =False)
    maxweight = weights[0]
    for w in weights:
      if w > maxweight:
        maxweight = w   
    acum_weight = 0
    acum_vec = np.zeros(6)
    weight_threshold = 0.7*maxweight
    for i in range(len(list_particles)):
      if weights[i] > weight_threshold:
        p = SE3.TranToVec(list_particles[i])
        acum_vec += p*weights[i]
        acum_weight += weights[i]
    estimated_particle = acum_vec*(1./acum_weight)
    bettermodel = SE3.VecToTran(estimated_particle)

    T_inv = np.linalg.inv(bettermodel)
    total_energy = 0.
    for datapoint in data:
      datapoint[0] = np.dot(T_inv[:3,:3],datapoint[0]) + T_inv[:3,3]
      datapoint[1] = np.dot(T_inv[:3,:3],datapoint[1])
      dist = ptcl.CalculateMahaDistanceMesh(woodstick,datapoint)
      print dist
      total_energy += dist**2
    err_thismodel = np.sqrt(total_energy)

    print "new err",err_thismodel
    print "Inliers indexes", maybeinliers_indexes
    if err_thismodel < besterr:
      besterr = err_thismodel
      bestfit = bettermodel
      bestindexes = maybeinliers_indexes
      print "Best err SO FAR", besterr
print 'Best indexes', bestindexes
print 'Best err', besterr
print "Resulting estimation:\n", bestfit
print "Real transformation\n", T



rawdata = copy.deepcopy(D)
list_particles, weights = ptcl.ScalingSeries(woodstick,ptcls0, rawdata, M, sigma0, sigma_desired,prune_percentage, dim,visualize = False)
maxweight = weights[0]
for w in weights:
  if w > maxweight:
    maxweight = w   
acum_weight = 0
acum_vec = np.zeros(6)
weight_threshold = 0.7*maxweight
for i in range(len(list_particles)):
  if weights[i] > weight_threshold:
    p = SE3.TranToVec(list_particles[i])
    acum_vec += p*weights[i]
    acum_weight += weights[i]
estimated_particle = acum_vec*(1./acum_weight)
transf = SE3.VecToTran(estimated_particle)
print "Resulting estimation w raw data:\n", transf
