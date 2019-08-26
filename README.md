COVARIANCE-BASED POSE ESTIMATION (COPE)
====================================

Author: [Huy Nguyen](https://sites.google.com/view/huy-nguyen/home)

This is cope, a library for working with uncertainty in pose estimation. It includes:

- **SO(3), SE(3) functionalities**.

- **Predicting the covariance of X in the AX=XB**.

- **Touch-based estimation in cluttered environment using particle filter**.

Requirements and Installation
-----------------------------

- cope requires Python >= 2.7. If you do not already have a Python environment configured on your computer, please see the instructions for installing the full [scientific Python stack](https://scipy.org/install.html).

- Touch-based estimation module requires *trimesh* for loading and using triangular meshes. To install *trimesh*, please see https://github.com/mikedh/trimesh

- To install the latest version of cope:
```bash    
   git clone https://github.com/dinhhuy2109/python-cope.git
   cd python-cope
   sudo python setup.py install
```


Basic usage and examples
------------

Please see the [wiki page](https://github.com/dinhhuy2109/python-cope/wiki) for more details.
