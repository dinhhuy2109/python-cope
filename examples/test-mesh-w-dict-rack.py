import trimesh
import numpy as np
import scipy as sp
import bisect
import SE3UncertaintyLib as SE3
import transformation as tr
import ParticleLib_offlineangle as ptcl
import pickle
import copy
import time
import IPython

def generate_measurements(mesh,pos_err,nor_err,num_measurements):
  ## Generate random points on obj surfaces
  # For individual triangle sampling uses this method:
  # http://mathworld.wolfram.com/TrianglePointPicking.html

  # # len(mesh.faces) float array of the areas of each face of the mesh
  # area = mesh.area_faces
  # # total area (float)
  # area_sum = np.sum(area)
  # # cumulative area (len(mesh.faces))
  # area_cum = np.cumsum(area)
  # face_pick = np.random.random(num_measurements)*area_sum
  # face_index = np.searchsorted(area_cum, face_pick)

  face_w_normal_up = []
  for i in range(len(mesh.faces)):
    if np.dot(mesh.face_normals[i],np.array((0,0,1))) >= -0.1:
      face_w_normal_up.append(i)
  face_index = np.random.choice(face_w_normal_up,num_measurements)
  # pull triangles into the form of an origin + 2 vectors
  tri_origins = mesh.triangles[:, 0]
  tri_vectors = mesh.triangles[:, 1:].copy()
  tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))
  # pull the vectors for the faces we are going to sample from
  tri_origins = tri_origins[face_index]
  tri_vectors = tri_vectors[face_index]
  # randomly generate two 0-1 scalar components to multiply edge vectors by
  random_lengths = np.random.random((len(tri_vectors), 2, 1))
  # points will be distributed on a quadrilateral if we use 2 0-1 samples
  # if the two scalar components sum less than 1.0 the point will be
  # inside the triangle, so we find vectors longer than 1.0 and
  # transform them to be inside the triangle
  random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
  random_lengths[random_test] -= 1.0
  random_lengths = np.abs(random_lengths)
  # multiply triangle edge vectors by the random lengths and sum
  sample_vector = (tri_vectors * random_lengths).sum(axis=1)
  # finally, offset by the origin to generate
  # (n,3) points in space on the triangle
  samples = sample_vector + tri_origins
  normals = mesh.face_normals[face_index]

  ## Transform points and add noise
  # point_errs = np.random.multivariate_normal(np.zeros(3),np.eye(3),num_measurements)
  random_vecs = np.random.uniform(-1,1,(num_measurements,3))
  point_errs = np.asarray([np.random.normal(0.,np.sqrt(3)*pos_err)*random_vec/np.linalg.norm(random_vec) for random_vec in random_vecs])
  noisy_points = copy.deepcopy(samples) + point_errs


  noisy_normals = [np.dot(tr.rotation_matrix(np.random.normal(0.,nor_err),np.cross(np.random.uniform(-1,1,3),n))[:3,:3],n) for n in normals]
  noisy_normals = np.asarray([noisy_n/np.linalg.norm(noisy_n) for noisy_n in noisy_normals])

  dist = [np.linalg.norm(point_err) for point_err in point_errs]
  alpha = [np.arccos(np.dot(noisy_normals[i],normals[i])) for i in range(len(normals))]
## not correct alpha err!!
  # print np.sqrt(np.cov(dist))
  # print np.sqrt(np.cov(alpha))
  measurements = [[noisy_points[i],noisy_normals[i]] for i in range(num_measurements)]
  return measurements #note that the normals here are sampled on obj surfaces


# pkl_file = open('mesh_w_dict.p', 'rb')
# mesh_w_dict = pickle.load(pkl_file)
# pkl_file.close()

# mesh = trimesh.load_mesh('featuretype.STL')

T_real = []
T_our = []
T_origin = []
time_our = []
time_origin = []
total_particle_our = []
total_perticle_origin = []
for i in range(5):
  
  mesh = trimesh.load_mesh('Rack1.ply')
  mesh.apply_translation(-mesh.centroid)
  pkl_file = open('rack_w_dict.p', 'rb')
  sorted_face = pickle.load(pkl_file)
  pkl_file.close()

  # Measurements' Errs
  pos_err = 2e-3
  nor_err = 5./180.0*np.pi

  num_measurements = 15
  measurements = generate_measurements(mesh,pos_err,nor_err,num_measurements)

  # # Visualize mesh and measuarement
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

  sigma0 = np.diag([0.0025, 0.0025,0.0025,0.25,0.25,0.25],0) #trans,rot
  # sigma0 = np.diag([0.0025, 0.0025,0.0025,1.6,1.6,1.6],0)
  # sigma0 = np.diag([0.0025, 0.0025,0.0025,1.,1.,1.],0)
  sigma_desired = .25*np.diag([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6],0)

  cholsigma0 = np.linalg.cholesky(sigma0).T
  uniformsample = np.random.uniform(-1,1,size = 6)
  xi_new_particle = np.dot(cholsigma0, uniformsample)
  T = SE3.VecToTran(xi_new_particle)
  # T = np.eye(4)
  for d in measurements:
      d[0] = np.dot(T[:3,:3],d[0]) + T[:3,3]
      d[1] = np.dot(T[:3,:3],d[1])

  dim = 6 # 6 DOFs
  prune_percentage = 0.8
  ptcls0 = [np.eye(4)]
  M = 8


  t0 = time.time()
  # Run scaling series
  list_particles, weights = ptcl.ScalingSeriesA(mesh,sorted_face, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)
  print 'running time', time.time() - t0
  time_our.append(time.time() - t0)

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
  T_real.append(T)
  T_our.append(transf)
  print "Resulting estimation:\n", transf
  print "Real transformation\n", T


  t0 = time.time()
  # Run scaling series
  list_particles, weights = ptcl.ScalingSeriesB(mesh,sorted_face, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)
  print time.time() - t0
  time_origin.append(time.time() - t0)
  
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
  T_origin.append(transf)
  print "Resulting estimation:\n", transf
  # print "Real transformation\n", T

pickle.dump([time_our,time_origin],open('time_back2.exp1','wb'))
pickle.dump([T_real,T_our,T_origin],open('T_back2.exp1','wb'))
IPython.embed()
