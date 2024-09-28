import numpy as np
import scipy
from scipy.stats import norm
from scipy.stats import multivariate_normal
import random

mean2 = [-1]
cov2  = [1.0] 
rv2 = norm(mean2, cov2)

def slope(p,z):
    dz = 0.001
    z0 = z-dz
    z1 = z+dz
    slope = (np.log(p.pdf(z1))-np.log(p.pdf(z0)))/(z1-z0)
    return slope

def envelope(z):
    out = 0.0
    if (z >= -5 and z <= -2):
        m = slope(rv2,-3.5)
        k = 1
        z0 = -3.5
        _out = k*(rv2.pdf(z0)+m*(z-z0))
        out  = np.exp(_out)
    if (z > -2 and z <= 2):
        m = slope(rv2,0)
        k = 1
        z0 = 0
        _out = k*(rv2.pdf(z0)+m*(z-z0))
        out  = np.exp(_out)
    if (z > 2 and z <= 5):
        m = slope(rv2,3.5)
        k = 1
        z0 = 3.5
        _out = k*(rv2.pdf(z0)+m*(z-z0))
        out  = np.exp(_out)
    return out