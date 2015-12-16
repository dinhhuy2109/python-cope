import numpy  as np
import COPE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

starttime = time.time()
tiny = 1e-5

T1 = np.eye(4)
sigma1 = np.diag([tiny,tiny,tiny,tiny,tiny,tiny],0)

T2 = np.eye(4)
T2[:3,3] = np.array([0,0.15,0])
sigma2 = np.diag([tiny,tiny,tiny,0.01,tiny,0.01],0)

T3 = np.eye(4)
T3[:3,3] = np.array([0,0,-0.03])
sigma3 = np.diag([tiny,tiny,tiny,tiny,tiny,0.1],0)

T4 = np.eye(4)
T4[:3,3] = np.array([0,0.14,0])
sigma4 = np.diag([tiny,tiny,tiny,tiny,tiny,tiny],0)



T12, sigma12 = COPE.Propagating(T1,sigma1,T2,sigma2)

T23, sigma23 = COPE.Propagating(T12,sigma12,T3,sigma3)

T34, sigma34 = COPE.Propagating(T23,sigma23,T4,sigma4)
print('Propagation took %.4f seconds' % (time.time() - starttime))

Tv = T34 
Tv[:3,3] = Tv[:3,3] + np.array([0.001,0.002,-0.001])
sigmav = np.diag([tiny,tiny,tiny,0.01,0.2,tiny])

Te,sigmae,iters = COPE.Fusing([T34,Tv],[sigma34,sigmav], maxiterations=10, retiter=True)

print('Took %.4f seconds and %d iterations' % (time.time() - starttime, iters))

COPE.Visualize([T12,T34,Te],[sigma12, sigma34,sigmae],100)
# visualize(T,sigma,100,2)


