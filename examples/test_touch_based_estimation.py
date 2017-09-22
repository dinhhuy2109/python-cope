import trimesh
import numpy as np
import cope.SE3lib as SE3
import cope.particlelib as ptcl
import pickle

extents    = [0.05,0.1,0.2]
mesh       = trimesh.creation.box(extents)
pkl_file   = open('data/woodstick_w_dict.p', 'rb')
angle_dict = pickle.load(pkl_file)
pkl_file.close()

measurements = [[np.array([-0.02882446, -0.04892219,  0.00738576]),
                 np.array([-0.40190523, -0.90828342, -0.11616118])],
                [np.array([ 0.01610016,  0.04007391,  0.01259628]),
                 np.array([ 0.52140577,  0.83554119,  0.17322511])],
                [np.array([ -1.470e-05,  2.2329e-02,  8.2384e-02]),
                 np.array([-0.88742601,  0.40497962,  0.22015128])],
                [np.array([-0.00351179, -0.05645598,  0.10106514]),
                 np.array([ 0.14455158, -0.24319869,  0.95914506])],
                [np.array([ 0.03399573,  0.01704082,  0.09381309]),
                 np.array([ 0.45363072,  0.87916087,  0.14592921])],
                [np.array([-0.03732133, -0.00669343, -0.01288346]),
                 np.array([-0.88147849,  0.34850243,  0.31865615])]]

real_T = np.array([[ 0.87960621,  0.45173617,  0.14908838, -0.00259753],
                   [-0.41013102,  0.87893701, -0.24343844, -0.00647988],
                   [-0.24100924,  0.15298419,  0.95838947,  0.00937599],
                   [ 0.        ,  0.        ,  0.        ,  1.        ]])
# Scaling series params
dim = 6 # 6 DOFs
prune_percentage = 0.8
ptcls0 = [np.eye(4)]
M = 6
# Measurements' Errs
pos_err = 2e-3
nor_err = 5./180.0*np.pi

sigma0 = np.diag([0.0025,0.0025,0.0025,0.25,0.25,0.25],0) #trans,rot
sigma_desired = np.diag([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6],0)

estimate = ptcl.RunScalingSeries(mesh,angle_dict, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)

print 'Estimated transformation:\n', estimate
print 'Real transformation:\n', real_T
