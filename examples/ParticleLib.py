#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Huy Nguyen <huy.nguyendinh09@gmail.com>
#
# This file is part of python-cope.
#
# python-cope is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# python-cope is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# python-cope. If not, see <http://www.gnu.org/licenses/>.

import numpy  as np
import trimesh as trm
import scipy as sp
import SE3UncertaintyLib as SE3lib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import copy
import IPython

def VisualizeParticles(list_particles,weights,env='',body ='', showestimated = False):
  maxweight = 0
  for w in weights:
    if w > maxweight:
      maxweight = w    
  if showestimated is False:
    weight_threshold = 0.7*maxweight
    from openravepy import RaveCreateKinBody
    with env:
      env.Reset()
      newbodies = []
      for i in range(len(list_particles)):
        if weights[i] > weight_threshold:
          p = list_particles[i]
          transparency = 0#1 - weights[i]/maxweight
          transf = p
          newbody = RaveCreateKinBody(env,body.GetXMLId())
          newbody.Clone(body,0)
          newbody.SetName(body.GetName())
          for link in newbody.GetLinks():
            for geom in link.GetGeometries():
              geom.SetTransparency(transparency)
              geom.SetDiffuseColor([0.64,0.35,0.14])
              env.AddKinBody(newbody,True)
          with env:
            newbody.SetTransform(transf)

          newbodies.append(newbody)
  else:
    acum_weight = 0
    acum_tf = np.zeros((4,4))
    weight_threshold = 0.7*maxweight
    for i in range(len(list_particles)):
      if weights[i] > weight_threshold:
        p = list_particles[i]
        acum_tf += p*weights[i]
        acum_weight += weights[i]
    estimated_particle = acum_tf*(1/acum_weight)
    transf = estimated_particle
    print "Resulting estimation:\n", transf
    body.SetTransform(transf)
    env.AddKinBody(body,True)
    return transf


class Region(object):
  def __init__(self, particles_list, sigma):
    self.particles_list = particles_list #List of particles (transformations)
    self.sigma = sigma

def EvenDensityCover(region, M):
  '''Input: Region V_n - sampling region represented as a union of neighborhoods, M - number of particles to sample per neighborhood
  Output: a set of particles that evenly cover the region (the new spheres will have analogous shape to the region sigma)
  '''
  particles = []
  num_spheres = len(region.particles_list)
  # num_total_particles = num_spheres*M
  sigma = region.sigma
  cholsigma = np.linalg.cholesky(sigma).T
  for i  in range(num_spheres):
    center_particle = region.particles_list[i]
    xi_center =  SE3lib.TranToVec(center_particle)
    for m in range(M):
      count = 0
      accepted = False
      while not accepted and count < 5:
        uniformsample = np.random.uniform(-1,1,size = 6)
        xi_new_particle = np.dot(cholsigma, uniformsample) + xi_center
        count += 1
        for k in range(i-1):
          previous_center = region.particles_list[k]
          if SE3lib.IsInside(xi_new_particle, SE3lib.TranToVec(previous_center),sigma):
            accepted = False
            break
        accepted = True
      if accepted:
        new_p = SE3lib.VecToTran(xi_new_particle)
        particles.append(new_p)
  # IPython.embed()
  return particles

class Mesh(object):
  def __init__(self, obj): #trimesh obj
    self.vertices = obj.vertices 
    self.faces = obj.faces
    self.normals = obj.face_normals
    self.facets = obj.facets

def ComputeNormalizedWeights(mesh, particles,measurements,tau):
  num_particles = len(particles)
  new_weights = np.zeros(num_particles)
  for i in range(len(particles)):
    T = np.linalg.inv(particles[i])
    D = copy.deepcopy(measurements)
    for d in D:
      d[0] = np.dot(T[:3,:3],d[0]) + T[:3,3]
      d[1] = np.dot(T[:3,:3],d[1])
    total_energy = sum([CalculateMahaDistanceMesh(mesh,d)**2 for d in D])
    new_weights[i] = (np.exp(-0.5*total_energy/tau))
  return normalize(new_weights)

def normalize(weights):
  norm_weights = np.zeros(len(weights))
  sum_weights = np.sum(weights)
  for i in range(len(weights)):
    norm_weights[i] = weights[i]/sum_weights
  return norm_weights

def CalculateMahaDistanceFace(face,d,i):
  '''
  :param face:     Vector [p1,p2,p3]: three points of a face
  :param d:           Measurement data [p,n,o_n,o_p]: measurement point and vector
  :return:
  '''
  p1,p2,p3,v = face
  p,n,o_p,o_n = d
  v = -v # reverse normal of the face
  # Get distance to surface
  norm = lambda x: np.linalg.norm(x)
  inner = lambda a, b: np.inner(a,b)
  diff_distance   = norm(inner((p-p1), v)/norm(v))
  # Get differences in normal direction
  diff_angle      = np.arccos(inner(v, n)/norm(v)/norm(n))
  # Maha distance
  ans = np.sqrt(diff_distance**2/o_p**2+diff_angle**2/o_n**2)
  return ans

def CalculateMahaDistanceMesh(mesh,d):
  '''
  :param  mesh:       A trimesh object
  :param  d   :       A measurement [p,n,o_n,o_p]
  '''
  dis = []
  for i in range(len(mesh.faces)):
    A,B,C = mesh.faces[i]
    dis.append(CalculateMahaDistanceFace([mesh.vertices[A],mesh.vertices[B],mesh.vertices[C],mesh.face_normals[i]],d,i))
  return min(dis)

def Pruning(list_particles, weights,percentage):
  assert (len(list_particles)==len(weights)),"Wrong input data, length of list of particles are not equal to length of weight"
  num_particles = len(list_particles)
  pruned_list = []
  new_list_p = []
  new_list_w = []
  c = [weights[0]]
  for i in range(num_particles-1):
    c.append(c[-1] + weights[i+1])
  u = [np.random.uniform(0,1)/num_particles]
  k = 0
  for i in range(num_particles):
    u[i] = u[0] + 1/num_particles*i
    while (u[i] > c[k]):
      k+=1
    new_list_p.append(list_particles[k])  
  for i in range(num_particles):
    if SameTransforamtions(new_list_p[i],new_list_p[i-1]):
      
    
def SameTransformations(T1,T2,rot_tol,trans_tol):
  return True

def Pruning_old(list_particles, weights,prune_percentage):
  assert (len(list_particles)==len(weights)),"Wrong input data, length of list of particles are not equal to length of weight"
  pruned_list = []
  maxweight = 0
  for w in weights:
    if w > maxweight:
      maxweight = w
  threshold = prune_percentage*maxweight
  for i in range(len(list_particles)):
    if weights[i] > threshold:
      pruned_list.append(list_particles[i])
  return pruned_list

def Visualize(list_particles=[],D=[]):
  show_ = mesh.copy()
  show_.apply_transform(list_particles[-1])
  color = np.array([  21, 51,  252, 255])
  for face in mesh.faces:
    mesh.visual.face_colors[face] = color
  for d in D:
    sphere = trm.creation.icosphere(3,0.0025)
    TF = np.eye(4)
    TF[:3,3] = d[0]
    TF2 = np.eye(4)
    angle = np.arccos(np.dot(d[1],np.array([0,0,1])))
    vec = np.cross(d[1],np.array([0,0,1]))
    TF2[:3,:3] = SE3lib.VecToRot(angle*vec)
    TF2[:3,3] = d[0] + np.dot(SE3lib.VecToRot(angle*vec),np.array([0,0,0.1/2.]))
    cyl = trm.creation.cylinder(0.001,0.1)
    cyl.apply_transform(TF2)
    show_ += cyl
    sphere.apply_transform(TF)
    show_ += sphere
  for z in range(len(list_particles)-1):
    new_mesh = mesh.copy()
    new_mesh.apply_transform(list_particles[z])
    show_ += new_mesh
    show_.show()
  return True


def ScalingSeries(mesh, V0, D, M, sigma0, sigma_desired, prune_percentage =0.6,dim = 6, visualize = False):
  """
  @type  V0:  ParticleFilterLib.Region
  @param V0:  initial uncertainty region
  @param  D:  a list of measurements [p,n,o_n,o_p] p is the contacted point, n is the approaching vector (opposite to normal)
  @param  M:  the no. of particles per neighborhood
  @param delta_desired: terminal value of delta
  @param dim: dimension of the state space (6 DOFs)
  """ 
  zoom = 2**(-1./dim)
  R, s , RT = np.linalg.svd(sigma0)
  Rd,sd, RTd = np.linalg.svd(sigma_desired)
  delta_0 = max(s)
  delta_desired = max(sd)
  volume_0 = np.pi**(dim/2)/sp.special.gamma(dim/2+1)*delta_0**dim
  volume_desired = np.pi**(dim/2)/sp.special.gamma(dim/2+1)*delta_desired**dim
  N = int(np.round(np.log2(volume_0/volume_desired)))
  print N
  uniform_weights = normalize(np.ones(len(V0.particles_list)))
  V_n = V0
  sigma_n = sigma0
  delta_n = delta_0
  particles = []
  weights = []
  t1 = 0.
  t2 = 0.
  t3 = 0.
  for n in range(N):
    tau =(delta_n/delta_desired)**2
    # Sample new set of particles based on from previous region and M
    t0 = time.time()
    particles = EvenDensityCover(V_n,M)
    print "len of new generated particles ", len(particles)
    print 'tau ', tau
    t1 += time.time() - t0
    t0 = time.time()
    # Compute normalized weights
    weights = ComputeNormalizedWeights(mesh, particles, D, tau)
    # print "weights after normalizing",  weights
    t2 += time.time() - t0 
    t0 = time.time()
    # Prune based on weights
    pruned_particles = Pruning(particles,weights,prune_percentage)
    t3 += time.time() - t0     
    print 'No. of particles, after pruning:', len(pruned_particles)
    # Create a new region from the set of particle left after pruning
    sigma_n = sigma_n*zoom
    R_n, s_n , RT_n = np.linalg.svd(sigma_n)
    delta_n = max(s_n)
    V_n = Region(pruned_particles,sigma_n)
    # raw_input()
    # print "delta_prv",  sigma
  print 't1 _ EVEN density', t1
  print 't2 _ UPDATE probability', t2
  print 't3 _ PRUNE particles', t3
  new_set_of_particles = EvenDensityCover(V_n,M)
  new_weights = ComputeNormalizedWeights(mesh,new_set_of_particles,D,1.0)
  return new_set_of_particles, new_weights
