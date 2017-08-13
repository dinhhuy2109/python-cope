from numpy import *
from openravepy import *
import COPE
from pylab import *
import time
import scipy.spatial
# import convexdecomposition


env = Environment()
env.SetViewer('qtcoin')
woodstick = env.ReadKinBodyXMLFile("woodstick.xml")
transparency = 0
for link in woodstick.GetLinks():
  for geom in link.GetGeometries():
    geom.SetTransparency(transparency)
    geom.SetDiffuseColor([0.64,0.35,0.14])
M = 1
ion()
ptcl0 = np.eye(4) 
sigma0 = np.diag([0.0009, 0.0009,0.0009,0.04,0.4,0.4])
V0 = COPE.Region([ptcl0], sigma0)

body=woodstick
list_particles = COPE.EvenDensityCover(V0, M)
start_time = time.time()
from openravepy import RaveCreateKinBody

bodies_indices = []
bodies_vertices = []
t_rave = 0
t_clone = 0
t_add = 0
t_settf = 0
with env:
  env.Reset()
  newbodies = []
  for i in range(len(list_particles)):
    p = list_particles[i]
    start_time = time.time()
    newbody = RaveCreateKinBody(env,body.GetXMLId())
    t_rave += time.time() - start_time

    start_time = time.time()
    newbody.Clone(body,0)
    t_clone += time.time() - start_time

    newbody.SetName(body.GetName())

    # start_time = time.time()
    # env.AddKinBody(newbody,True)
    # t_add += time.time() - start_time

    start_time = time.time()
    newbody.SetTransform(p)
    t_settf += time.time() - start_time
    trimesh = env.Triangulate(newbody)
    bodies_indices.append(trimesh.indices)
    bodies_vertices.append(trimesh.vertices)
    newbodies.append(newbody)

# np.vstack(bodies_indices)
# points = np.vstack(bodies_vertices)
# hull = scipy.spatial.ConvexHull(points)
# hull = scipy.spatial.ConvexHull(points[hull.vertices])

# bodies_mesh =  TriMesh(hull.points, hull.simplices)
# newkinbody = RaveCreateKinBody(env, '')
# newkinbody.InitFromTrimesh(bodies_mesh)
# newkinbody.SetName('convexkinbody')
# env.AddKinBody(newkinbody)
print t_rave, "s RaveCreateKinbody"
print t_clone, "s Clone"
print t_add, "s Add"
print t_settf, "s Settf"
raw_input()
