import torch
import torch.nn as nn
import numpy as np
import pyro
import random
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

def true_output(x):
    return (10*x+2)

def noisy_obs(x_lo, x_hi, Np, sigma):
    x_list=[]
    t_list=[]
    for i in range (0,Np):
        x = random.uniform(x_lo,x_hi)
        t = true_output(x)+random.gauss(0,sigma)
        x_list.append(x)
        t_list.append(t)
    return(np.array(x_list), np.array(t_list))

def UniGaussian(mu,std, array):
    t1  = (2*np.pi*std**2)**-0.5
    t2  = (array-mu)**2/(2*std**2)
    return(t1*np.exp(-t2))