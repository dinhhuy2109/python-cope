import numpy as np
import cope
import cope.transformation as tr


ksamples = 500
T =  tr.random_rotation_matrix()
T[:3,3] = tr.random_vector(3)
vec = cope.TranToVec(T)
scale = 1e-3
sigma = scale*np.diag((0.1,0.2,0.5,0.5,0.2,0.3))

vec_Tinv = cope.TranToVec(np.linalg.inv(T))

xi_vec_Tinv = []
for i in range(ksamples):
    xisample = np.random.multivariate_normal(np.zeros(6),sigma)
    Tsample = np.dot(cope.VecToTran(xisample),T)
    Tinvsample = np.linalg.inv(Tsample)
    xi_vec_Tinv.append(cope.TranToVec(np.dot(Tinvsample,T)))
    
print np.cov(np.transpose(xi_vec_Tinv))[:3,:3]
print cope.CovInverseTran(T,sigma)[1][:3,:3]
print cope.CovInverseTran(T,sigma)[1][3:,3:]
print np.cov(np.transpose(xi_vec_Tinv))[3:,3:]
