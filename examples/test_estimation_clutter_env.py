import trimesh
import numpy as np
import cope.SE3lib as SE3
import cope.particlelib as ptcl
import cope.transformation as tr
import copy
import time
import pickle


extents    = [0.13,0.1,0.3]
mesh       = trimesh.creation.box(extents)
pkl_file   = open('data/woodstick_w_dict.p', 'rb')
angle_dict = pickle.load(pkl_file)
pkl_file.close()

# ext = [0.05,0.05,0.34]
# other_box = trimesh.creation.box(ext)
# other_box.apply_translation([0.1,0.01,.02])

# rack = trimesh.load_mesh('data/Rack1.ply')
# rack.apply_translation([-0.163,-0.01,-0.17])
# rack.apply_transform(tr.euler_matrix(-3.14/5.,0,3.14/6.))
# rack.apply_translation([0.075,-0.075,0])

# clutter = copy.deepcopy(other_box + rack)


pos_err = 2e-3
nor_err = 5./180.0*np.pi

# Uncertainty & params
sigma0 = np.diag([0.0001,0.0001,0.0001,0.09,0.09,0.09],0)
sigma_desired = 0.09*np.diag([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6],0)
dim = 6 # 6 DOFs
prune_percentage = 0.8
ptcls0 = [np.eye(4)]
M = 6

measurements = [[np.array([-0.06538186,  0.00749609, -0.08090193]),
                 np.array([-0.96346864,  0.26295561, -0.05081864])],
                [np.array([ 0.04767954,  0.06771935, -0.09514227]),
                 np.array([ 0.13469883,  0.97865803,  0.15519239])],
                [np.array([ 0.06556443,  0.03302991,  0.0257163 ]),
                 np.array([ 0.98789178, -0.15154798,  0.03321196])],
                [np.array([-0.0762085 , -0.01166338,  0.14179293]),
                 np.array([-0.18657649, -0.19089539,  0.96371581])],
                [np.array([ 0.01253209,  0.03493477,  0.14190642]),
                 np.array([ 0.09914534,  0.9884659 ,  0.11447864])],
                [np.array([ 0.0503946 , -0.03325389,  0.05182237]),
                 np.array([ 0.98155197, -0.17806332,  0.06963612])],
                [np.array([ 0.00215572,  0.00585409,  0.15343024]),
                 np.array([-0.12381661, -0.10678936,  0.98654218])],
                [np.array([ 0.04817958, -0.02643583, -0.14640571]),
                 np.array([-0.1449615 , -0.94534233, -0.29208566])],
                [np.array([-0.05372821,  0.04857448, -0.11539388]),
                 np.array([-0.97851842,  0.18253174, -0.09583247])],
                [np.array([ 0.04072001,  0.05814159, -0.02821377]),
                 np.array([ 0.1465636 ,  0.96913464,  0.1982351 ])],
                [np.array([ 0.02121399, -0.0627091 ,  0.14400655]),
                 np.array([-0.13638759, -0.15341598,  0.97870423])],
                [np.array([ 0.01341456, -0.06184187,  0.09692748]),
                 np.array([-0.27290447, -0.94127916, -0.19878804])],
                [np.array([-0.08325116, -0.00788009,  0.10997277]),
                 np.array([-0.97692982,  0.18641623, -0.10419747])],
                [np.array([ 0.13258646, -0.05435934,  0.01305379]),
                 np.array([ 0.19554626, -0.84950123,  0.4900095 ])],
                [np.array([-0.05478498, -0.15888366, -0.0831765 ]),
                 np.array([ 0.19554626, -0.84950123,  0.4900095 ])]]

T = np.array([[  9.74890642e-01,   1.97848726e-01,  -1.02196470e-01, -4.01176998e-03],
              [ -2.11556189e-01,   9.66141266e-01,  -1.47699132e-01,  1.00540901e-02],
              [  6.95141413e-02,   1.65610798e-01,   9.83738201e-01, -1.46057980e-04],
              [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,  1.00000000e+00]])


# RANSAC
n = 5  #  the minimum number of data values required to fit the model
k = 50 # the maximum number of iterations allowed in the algorithm
threshold = 3.  # a threshold value for determining when a data point fits a model
d = 7  # the number of good data values required to assert that a model fits well to data

t0 = time.time()
ransac_transformation, ransac_score, ransac_inliers_idx = ptcl.RansacParticle(n,k,threshold,d,mesh,angle_dict, ptcls0, measurements, pos_err, nor_err, M, sigma0, sigma_desired, prune_percentage,dim = 6, visualize = False)
print 'Time', time.time() -t0
print 'Ransac transformation\n', ransac_transformation
print "Real transformation\n", T
T_r = ransac_transformation
print 'Dist trans:'
print np.linalg.norm(T_r[:3,3]-T[:3,3])
print 'Dist rot:'
print np.linalg.norm(SE3.RotToVec(np.dot(np.linalg.inv(T_r[:3,:3]),T[:3,:3])))
print 'Number of Inliers Detected:', len(ransac_inliers_idx)
detected_inliers = [measurements[idx] for idx in ransac_inliers_idx] 
ptcl.Visualize(mesh,T_r,measurements)
