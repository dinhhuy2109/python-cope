import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

import COPE as cope
import transformation as tr

print(__doc__)
import time

# Number of samples per component
n_samples = 5000

# Generate random sample, two components
# True seed
Ra_true = tr.random_rotation_matrix()[:3,:3]
alpha_true = cope.RotToVec(Ra_true)
ta_true = np.random.random_sample(3)
sigmaRa = 1e-5*np.diag((4 , 9, 3))
sigmata = 1e-5*np.diag((5 , 7, 1))

Rb_true = tr.random_rotation_matrix()[:3,:3]
beta_true =  cope.RotToVec(Rb_true)
tb_true = np.random.random_sample(3)
sigmaRb = 1e-5*np.diag((3 , 6, 8))
sigmatb = 1e-5*np.diag((8 , 1, 2))

Rot = np.r_[np.random.multivariate_normal(alpha_true,sigmaRa,n_samples),np.random.multivariate_normal(beta_true,sigmaRb,n_samples)]

Trans=np.r_[np.random.multivariate_normal(ta_true,sigmata,n_samples),np.random.multivariate_normal(tb_true,sigmatb,n_samples)]
start_time = time.time()
#GMM for ROT
lowest_bic = np.infty
bic = []
n_components_range = range(1, 10)
cv_types = ['diag']#['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(Rot)
        bic.append(gmm.bic(Rot))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
# bic = np.array(bic)
best_rot_gmm = best_gmm
t_fit = time.time() - start_time
print t_fit, 's'

start_time2 = time.time()
#GMM for TRANS
lowest_bic = np.infty
bic = []
n_components_range = range(1,10)
cv_types = ['diag']#['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(Trans)
        bic.append(gmm.bic(Trans))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
# bic = np.array(bic)
best_trans_gmm = best_gmm

t_fit = time.time() - start_time2
print t_fit, 's'

# TRANS AND ROT (vec of 6 elements)x

A_true = np.r_[ta_true,alpha_true]#np.random.random_sample(6)
sigmaA = 1e-5*np.diag((4 , 9, 3, 5 , 7, 1))

B_true = np.r_[tb_true,beta_true] # np.random.random_sample(6)
sigmaB = 1e-5*np.diag((3,6,8,8,1,2))

X = np.r_[np.random.multivariate_normal(A_true,sigmaA,n_samples),np.random.multivariate_normal(B_true,sigmaB,n_samples)]

start_time = time.time()
lowest_bic = np.infty
bic = []
n_components_range = range(1, 10)
cv_types = ['diag']#['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
# bic = np.array(bic)
best_X_gmm = best_gmm

t_fit = time.time() - start_time
print t_fit, 's'

print best_X_gmm.covariances_
