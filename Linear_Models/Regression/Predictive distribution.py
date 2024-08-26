# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import random
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
import functions
import matplotlib as mpl
from matplotlib import  ticker

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


# +
def basis (x):
    temp = [1, x, x**2, x**3, x**4, x**5]
    return(np.array(temp))

def true_generator(x):
    return (25*(x-0.5)**3)

def noisy_generator(Np,alpha):
    x_out = []
    t_out = []
    for i in range(0,Np):
        x = random.uniform(0,1)
        temp1 = true_generator(x)
        temp2 = temp1+random.gauss(0,alpha)
        x_out.append(x)
        t_out.append(temp2)
    return(x_out, t_out)


# +
beta  = 0.5
alpha = 0.2

x_sam = np.linspace(0,1,100)
phi = functions.ModelMatrix(np.size(x_sam),x_sam)
# -

# ## Observes 2 points

# +
x_obs1, t_obs1 = noisy_generator(2,beta)

Phi = functions.ModelMatrix(np.size(x_obs1),x_obs1)
SN_inv =  alpha*np.identity(6)+beta*np.matmul(Phi.T,Phi)

SN = np.linalg.inv(SN_inv)

mN = np.matmul(beta*SN, np.matmul(Phi.T,t_obs1))

# +
pred_mean = []
pred_std  = []
for i in range (0,100):
    x = x_sam[i]
    p_mean = np.matmul(mN.T, basis(x))
    pred_mean.append(p_mean)
    p_std2 = np.matmul(np.matmul(basis(x).T,SN),basis(x))+beta
    pred_std.append(np.sqrt(p_std2))

pred_mean= np.array(pred_mean)
pred_std = np.array(pred_std)

# +
plt.plot(x_sam, pred_mean,'r')
plt.plot(x_obs1,t_obs1, 'ko', mfc='w',markersize=8)
plt.plot(x_sam, true_generator(x_sam),'b')
plt.fill_between(x_sam, pred_mean-pred_std, pred_mean+pred_std,alpha=0.2,color='r')
plt.xlim(0,1); plt.ylim(-5,5)

plt.xlabel('$x$',fontsize=22)
plt.ylabel('$t$',fontsize=22)
plt.savefig('predictive_dist_1.png')

# +
## Observes 10 points

# +
x_obs1, t_obs1 = noisy_generator(10,beta)

Phi = functions.ModelMatrix(np.size(x_obs1),x_obs1)
SN_inv =  alpha*np.identity(6)+beta*np.matmul(Phi.T,Phi)

SN = np.linalg.inv(SN_inv)

mN = np.matmul(beta*SN, np.matmul(Phi.T,t_obs1))

# +
pred_mean = []
pred_std  = []
for i in range (0,100):
    x = x_sam[i]
    p_mean = np.matmul(mN.T, basis(x))
    pred_mean.append(p_mean)
    p_std2 = np.matmul(np.matmul(basis(x).T,SN),basis(x))+beta
    pred_std.append(np.sqrt(p_std2))

pred_mean= np.array(pred_mean)
pred_std = np.array(pred_std)

# +
plt.plot(x_sam, pred_mean,'r')
plt.plot(x_obs1,t_obs1, 'ko', mfc='w',markersize=8)
plt.plot(x_sam, true_generator(x_sam),'b')
plt.fill_between(x_sam, pred_mean-pred_std, pred_mean+pred_std,alpha=0.2,color='r')
plt.xlim(0,1); plt.ylim(-5,5)

plt.xlabel('$x$',fontsize=22)
plt.ylabel('$t$',fontsize=22)
plt.savefig('predictive_dist_2.png')

# +
## Observes 200 points

# +
x_obs1, t_obs1 = noisy_generator(200,beta)

Phi = functions.ModelMatrix(np.size(x_obs1),x_obs1)
SN_inv =  alpha*np.identity(6)+beta*np.matmul(Phi.T,Phi)

SN = np.linalg.inv(SN_inv)

mN = np.matmul(beta*SN, np.matmul(Phi.T,t_obs1))

# +
pred_mean = []
pred_std  = []
for i in range (0,100):
    x = x_sam[i]
    p_mean = np.matmul(mN.T, basis(x))
    pred_mean.append(p_mean)
    p_std2 = np.matmul(np.matmul(basis(x).T,SN),basis(x))+beta
    pred_std.append(np.sqrt(p_std2))

pred_mean= np.array(pred_mean)
pred_std = np.array(pred_std)

# +
plt.plot(x_sam, pred_mean,'r')
plt.plot(x_obs1,t_obs1, 'ko', mfc='w',markersize=6)
plt.plot(x_sam, true_generator(x_sam),'b')
plt.fill_between(x_sam, pred_mean-pred_std, pred_mean+pred_std,alpha=0.2,color='r')
plt.xlim(0,1); plt.ylim(-5,5)

plt.xlabel('$x$',fontsize=22)
plt.ylabel('$t$',fontsize=22)
plt.savefig('predictive_dist_3.png')
# -


