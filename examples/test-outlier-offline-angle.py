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
   list_particles, weights = ptcl.ScalingSeriesB(mesh,sorted_face, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)
   
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
  print dist, threshold
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
    # raw_input()
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

pos_err_outlier = 50e-3
nor_err_outlier = 50./180.0*np.pi
num_outliers = 3
outliers = ptcl.GenerateMeasurements(mesh,pos_err_outlier,nor_err_outlier,num_outliers)

measurements = copy.deepcopy(inliers + outliers)
measurements = [[np.array([ 0.02020934, -0.00093154, -0.04818443]),
  np.array([ 0.99092475, -0.10126434,  0.08839504])],
 [np.array([ 0.02230125, -0.00838275, -0.0544496 ]),
  np.array([ 0.99268845, -0.02376552,  0.11834209])],
 [np.array([-0.00044406,  0.00389368,  0.05819387]),
  np.array([ 0.08800807,  0.9636131 ,  0.25239725])],
 [np.array([-0.03384596, -0.01920963,  0.17853665]),
  np.array([-0.02158618, -0.12763422,  0.99158638])],
 [np.array([-0.02859522, -0.01616938,  0.12176566]),
  np.array([-0.07438778, -0.98612014, -0.14843697])],
 [np.array([-0.00488758, -0.00768086, -0.02598989]),
  np.array([-0.02095794, -0.99877968, -0.04472045])],
 [np.array([-0.02021277,  0.02075643, -0.1498303 ]),
  np.array([ 0.14026719,  0.97193785,  0.18884369])],
 [np.array([-0.02059115, -0.02535577,  0.18090993]),
  np.array([-0.0449068 ,  0.00590515,  0.99897373])],
 [np.array([ -3.89635169e-02,  -2.66610422e-05,   1.07404905e-01]),
  np.array([-0.98686551,  0.15952528,  0.0254587 ])],
 [np.array([ 0.01949362,  0.00311245,  0.05224403]),
  np.array([ 0.06794772,  0.99447112,  0.08006432])],
 [np.array([ 0.04585108, -0.03629538, -0.10581397]),
  np.array([-0.41481841, -0.90695784, -0.07316524])],
 [np.array([-0.05054189, -0.04681321, -0.12583244]),
  np.array([ 0.45212409, -0.88942588,  0.06712228])],
 [np.array([-0.03489201, -0.02910762,  0.17616147]),
  np.array([-0.8131653 , -0.31389047, -0.49013771])]]
T = np.array([[ 0.99674075,  0.05832855, -0.05572835, -0.00781362],
       [-0.06259863,  0.99496818, -0.07822868, -0.00166119],
       [ 0.05088497,  0.08146223,  0.99537662,  0.01204734],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
  
# Visualize mesh and measuarements
# color = trimesh.visual.random_color()
# for face in mesh.faces:
#     mesh.visual.face_colors[face] = color
# show = mesh.copy()
# color = trimesh.visual.random_color()
# for d in measurements:
#   sphere = trimesh.creation.icosphere(3,0.005)
#   TF = np.eye(4)
#   TF[:3,3] = d[0]
#   sphere.apply_transform(TF)
#   show+=sphere
# show.show()

# Uncertainty & params
# # sigma0 = np.diag([0.0009, 0.0009,0.0009,1.,1.,1.],0)
sigma0 = np.diag([0.0009, 0.0009,0.0009,0.01,0.01,0.01],0)
sigma_desired = 0.25*np.diag([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6],0)
# cholsigma0 = np.linalg.cholesky(sigma0).T
# uniformsample = np.random.uniform(-1,1,size = 6)
# xi_new_particle = np.dot(cholsigma0, uniformsample)
# T = SE3.VecToTran(xi_new_particle)

# for d in measurements:
#     d[0] = np.dot(T[:3,:3],d[0]) + T[:3,3]
#     d[1] = np.dot(T[:3,:3],d[1])
dim = 6 # 6 DOFs
prune_percentage = 0.7
ptcls0 = [np.eye(4)]
M = 10

# IPython.embed()

# Run Scaling series with all measurements
all_measurements_transf =  RunScalingSeries(mesh,sorted_face, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)

# RANSAC
n = 5  #  the minimum number of data values required to fit the model
k = 10 # the maximum number of iterations allowed in the algorithm
threshold = 3.  # a threshold value for determining when a data point fits a model
d = 7  # the number of good data values required to assert that a model fits well to data
ransac_transformation, ransac_score, ransac_inliers_idx = RansacParticle(n,k,threshold,d,mesh,sorted_face, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)

print 'Ransac transformation', ransac_transformation
print 'best idx' ,ransac_inliers_idx
print 'best score', ransac_score
IPython.embed()
# If use all the measurements
print "Resulting estimation using all measurements:\n", all_measurements_transf
print "Real transformation\n", T
