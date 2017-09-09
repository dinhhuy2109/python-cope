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
import time

def RunScalingSeries(mesh,sorted_face, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False):
   list_particles, weights = ptcl.ScalingSeriesA(mesh,sorted_face, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)
   
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
  # print dist, threshold
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
    print maybeinliers_idx, 'len', len(maybeinliers_idx)
    if len(maybeinliers) > d: # Maybe good hypothesis
      print 'Maybe good hypothesis'
      updated_hypothesis = RunScalingSeries(mesh,sorted_face, ptcls0, maybeinliers, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)
      score = ScoreHypothesis(updated_hypothesis,maybeinliers,pos_err,nor_err,mesh,sorted_face)
      if score > best_score:
        best_hypothesis = updated_hypothesis
        best_score = score
        best_idx = maybeinliers_idx
        if score > 8e-5:
            break
        print iterations
    # raw_input()
  return best_hypothesis,best_score,best_idx

def GenerateMeasurementsInlierOutlier(mesh,initial_inliers):
  inliers = []
  outliers = []
  for measurement in initial_inliers:
    intersect = mesh.ray.intersects_location([measurement[0]],[measurement[1]])
    if len(intersect[0]) > 1:
      # replace by outlier
      dist = [np.linalg.norm(intersect_point-measurement[0]) for intersect_point in intersect[0]]
      max_dist_idx = dist.index(max(dist))
      point = intersect[0][max_dist_idx]
      normal = mesh.face_normals[intersect[2][max_dist_idx]]
      outliers.append([point,normal])
    else:
      inliers.append(measurement)
  return inliers,outliers




pkl_file = open('woodstick_w_dict.p', 'rb')
sorted_face = pickle.load(pkl_file)
pkl_file.close()

extents = [0.13,0.1,0.3]
mesh = trimesh.creation.box(extents)

color = np.array([252,   2,  92, 255])# trimesh.visual.random_color()
for facet in mesh.facets:
    mesh.visual.face_colors[facet] = color
for face in mesh.faces:
    mesh.visual.face_colors[face] = color

ext = [0.05,0.05,0.34]
other_box = trimesh.creation.box(ext)
other_box.apply_translation([0.1,0.01,.02])

color2 = np.array([134,   2, 252, 255])#trimesh.visual.random_color()
for facet in other_box.facets:
    other_box.visual.face_colors[facet] = color2
for face in other_box.faces:
    other_box.visual.face_colors[face] = color2

rack = trimesh.load_mesh('Rack1.ply')
rack.apply_translation([-0.163,-0.01,-0.17])
rack.apply_transform(tr.euler_matrix(-3.14/5.,0,3.14/6.))
rack.apply_translation([0.075,-0.075,0])

color3 = np.array([  2, 252, 177, 255])#trimesh.visual.random_color()#np.array([  2, 252,  12, 255]
for facet in rack.facets:
    rack.visual.face_colors[facet] = color3
for face in rack.faces:
    rack.visual.face_colors[face] = color3

clutter = copy.deepcopy(mesh + other_box + rack)

# clutter.show()
# IPython.embed()
# raw_input()

# Measurements' Errs
pos_err = 2e-3
nor_err = 5./180.0*np.pi
num_measurements = 15
initial_inliers = ptcl.GenerateMeasurementsTriangleSampling(mesh,pos_err,nor_err,num_measurements)

inliers,outliers = GenerateMeasurementsInlierOutlier(clutter,initial_inliers)

print "Num inliers/Num outliers: ", len(inliers),'/',len(outliers)
measurements = copy.deepcopy(inliers+outliers)
  
# Visualize mesh and measuarements
show = clutter.copy()
for d in measurements:
  sphere = trimesh.creation.icosphere(3,0.005)
  TF = np.eye(4)
  TF[:3,3] = d[0]
  sphere.apply_transform(TF)
  show+=sphere
show.show()

# Uncertainty & params
sigma0 = np.diag([0.0025,0.0025,0.0025,0.25,0.25,0.25],0)
# sigma0 = np.diag([0.0009, 0.0009,0.0009,0.01,0.01,0.01],0)
sigma_desired = 0.64*np.diag([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6],0)
cholsigma0 = np.linalg.cholesky(sigma0).T
uniformsample = np.random.uniform(-1,1,size = 6)
xi_new_particle = np.dot(cholsigma0, uniformsample)
T = SE3.VecToTran(xi_new_particle)

for d in inliers:#measurements:
    d[0] = np.dot(T[:3,:3],d[0]) + T[:3,3]
    d[1] = np.dot(T[:3,:3],d[1])
dim = 6 # 6 DOFs
prune_percentage = 0.8
ptcls0 = [np.eye(4)]
M = 6

for d in measurements:
    d[0] = np.dot(T[:3,:3],d[0]) + T[:3,3]
    d[1] = np.dot(T[:3,:3],d[1])

# Run Scaling series with all measurements
t0 = time.time()
all_measurements_transf =  RunScalingSeries(mesh,sorted_face, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)
# print all_measurements_transf
# print T
print 'Time', time.time() -t0
# IPython.embed()

# RANSAC
n = 5  #  the minimum number of data values required to fit the model
k = 50 # the maximum number of iterations allowed in the algorithm
threshold = 3.  # a threshold value for determining when a data point fits a model
d = 7  # the number of good data values required to assert that a model fits well to data

t0 = time.time()
ransac_transformation, ransac_score, ransac_inliers_idx = RansacParticle(n,k,threshold,d,mesh,sorted_face, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)
print 'Ransac transformation', ransac_transformation
print 'best idx' ,ransac_inliers_idx
print 'best score', ransac_score
print 'Time', time.time() -t0

# If use all the measurements
print "Resulting estimation using all measurements:\n", all_measurements_transf
print "Real transformation\n", T

IPython.embed()
