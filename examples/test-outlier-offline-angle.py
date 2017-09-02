import trimesh
import numpy as np
import scipy as sp
import random
import SE3UncertaintyLib as SE3
import transformation as tr
import ParticleLib_offlineangle as ptcl
import pickle
import copy
import IPython

def RunScalingSeries(mesh,sorted_face, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False):
   ptcl.ScalingSeriesB(mesh,sorted_face, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)
   
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
   return SE3.VecToTran(estimated_particle)

def MeasurementFitHypothesis(hypothesis,measurement,pos_err,nor_err,mesh,sorted_face,distance_threshold):
  d = copy.deepcopy(measurement)
  T_inv = np.linalg.inv(hypothesis)
  d[0] = np.dot(T_inv[:3,:3],d[0]) + T_inv[:3,3]
  d[1] = np.dot(T_inv[:3,:3],d[1])
  dist = ptcl.FindminimumDistanceMeshOriginal(mesh,sorted_face,d,pos_err,nor_err)
  if dist < threshold:
    return True
  else: 
    return False

def ScoreHypothesis(hypothesis,measurements,pos_err,nor_err,mesh,sorted_face):
  data = copy.deepcopy(measurements)
  T_inv = np.linalg.inv(hypothesis)
  dist = 0.
  for d in data:
    d[0] = np.dot(T_inv[:3,:3],d[0]) + T_inv[:3,3]
    d[1] = np.dot(T_inv[:3,:3],d[1])
  dist = sum([ptcl.FindminimumDistanceMeshOriginal(mesh,sorted_face,d,pos_err,nor_err)**2 for d in data])
  score = np.exp(-0.5*dist)
  return score

def RansacParticle(n,k,threshold,d,mesh,sorted_face, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False):
  iterations = 0
  best_hypothesis = np.eye(4) # SHOULD BE ptcl0
  best_score = 0.
  num_measurements = len(measurements)
  while iterations < k:
    iterations += 1
    maybeinliers_idx = random.sample(range(num_measurements),n)
    maybeinliers = [measurements[i] for i in maybeinliers_idx]
    
    alsoinliers = []
    hypothesis = RunScalingSeries(mesh,sorted_face, ptcls0, maybeinliers, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)
    
    for i in range(len(measurements)):
      if i not in maybeinliers_idx:
        if MeasurementFitHypothesis(hypothesis,measurements[i],pos_err,nor_err,mesh,sorted_face,threshold):
          maybeinliers.append(measurements[i])
          maybeinliers_idx.append(i)
    if len(maybeinliers) > d: # Maybe good hypothesis
      updated_hypothesis = RunScalingSeries(mesh,sorted_face, ptcls0, maybeinliers, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)
      score = ScoreHypothesis(updated_hypothesis,maybeinliers,pos_err,nor_err,mesh,sorted_face)
      if score > best_score:
        best_hypothesis = updated_hypothesis
        best_score = score
        best_idx = maybeinliers_idx
  return best_hypothesis,best_score,best_idx


pkl_file = open('woodstick_w_dict.p', 'rb')
sorted_face = pickle.load(pkl_file)
pkl_file.close()

extents = [0.05,0.02,0.34]
mesh = trimesh.creation.box(extents)

# Measurements' Errs
pos_err = 2e-3
nor_err = 5./180.0*np.pi
num_inliers = 10
inliers = ptcl.GenerateMeasurements(mesh,pos_err,nor_err,num_inliers)

pos_err_outlier = 15e-3
nor_err_outlier = 20./180.0*np.pi
num_outliers = 5
outliers = ptcl.GenerateMeasurements(mesh,pos_err_outlier,nor_err_outlier,num_outliers)

measurements = copy.deepcopy(inliers + outliers)

# Visualize mesh and measuarements
color = trimesh.visual.random_color()
for face in mesh.faces:
    mesh.visual.face_colors[face] = color
show = mesh.copy()
color = trimesh.visual.random_color()
for d in measurements:
  sphere = trimesh.creation.icosphere(3,0.005)
  TF = np.eye(4)
  TF[:3,3] = d[0]
  sphere.apply_transform(TF)
  show+=sphere
show.show()

# Uncertainty & params
sigma0 = np.diag([0.0009, 0.0009,0.0009,1.,1.,1.],0)
sigma_desired = 0.25*np.diag([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6],0)
cholsigma0 = np.linalg.cholesky(sigma0).T
uniformsample = np.random.uniform(-1,1,size = 6)
xi_new_particle = np.dot(cholsigma0, uniformsample)
T = SE3.VecToTran(xi_new_particle)
# T = np.eye(4)
for d in measurements:
    d[0] = np.dot(T[:3,:3],d[0]) + T[:3,3]
    d[1] = np.dot(T[:3,:3],d[1])
dim = 6 # 6 DOFs
prune_percentage = 0.7
ptcls0 = [np.eye(4)]
M = 10

# IPython.embed()

# Run Scaling series with inliers
list_particles, weights = ptcl.ScalingSeriesB(mesh,sorted_face, ptcls0, measurements[:num_inliers], pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)
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
print "Resulting estimation using inliers:\n", transf
print "Real transformation\n", T

# RANSAC
n = 5  #  the minimum number of data values required to fit the model
k = 10 # the maximum number of iterations allowed in the algorithm
threshold = 2  # a threshold value for determining when a data point fits a model
d = 7  # the number of good data values required to assert that a model fits well to data
ransac_transformation, ransac_score, ransac_inliers_idx = RansacParticle(n,k,threshold,d,mesh,sorted_face, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)

print 'Ransac transformation', ransac_transformation
