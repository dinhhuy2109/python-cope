COVARIANCE-BASED POSE ESTIMATION (COPE)
====================================

Author: Huy Nguyen

Email:  huy.nguyendinh09@gmail.com

This library includes tools for working in **the space of rigid-body motions SE(3) and the space of rotation group SO(3)**, including many mathematic operators using Lie Group and Lie algebra.

It also includes tools for manipulating uncertainty of three-dimensional poses: **Propagating and fusing uncertainty in SE(3)** (implemented by following ""Timothy D Barfoot and Paul T Furgale, Associating Uncertainty with Three-Dimensional Poses for use in Estimation Problems""). In addition, we extended this to the case where poses is seperated into rotation and translation. 

Developing: Reducing uncertainty of object pose via touch sensing using particle filter. (best next touch & pose estimation) 

Requirements and Installation
-----------------------------

- Clone this folder, then go to that repository and run

   $ sudo python setup.py install


Examples
------------

