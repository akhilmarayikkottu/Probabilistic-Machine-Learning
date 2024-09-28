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

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from scipy.stats import multivariate_normal
import random

# +
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

img_dir = 'Images/'

# +
mean1 = [0]
cov1  = [2.0] 

mean2 = [-1]
cov2  = [1.0] 

k     = 2.5 
Np    = 500
# -

rv1 = norm(mean1, cov1)
rv2 = norm(mean2, cov2)

# +
x = np.linspace(-5,5,1000)

plt.figure(figsize=(6,4))
plt.plot(x,k*rv1.pdf(x),'r', label='$k q(z)$')
plt.plot(x,rv2.pdf(x),'b',label='$\\tilde{p}(z)$')
plt.fill_between(x, k*rv1.pdf(x), rv2.pdf(x),color='gray',alpha=0.5)
plt.xlim(-5,5);plt.ylim(-0.05,0.55)
plt.xlabel('$z$',fontsize=18)
plt.ylabel('$p$', fontsize=18)
plt.legend(frameon=False,fontsize=18)

plt.savefig(img_dir+'config.png')

# +
plt.figure(figsize=(6,4))
plt.plot(x,k*rv1.pdf(x),'r', label='$k q(z)$',alpha=0.4)
plt.plot(x,rv2.pdf(x),'b',label='$\\tilde{p}(z)$')
plt.fill_between(x, k*rv1.pdf(x), rv2.pdf(x),color='gray',alpha=0.4)
plt.xlim(-5,5);plt.ylim(-0.05,0.55)
plt.xlabel('$z$',fontsize=18)
plt.ylabel('$p$', fontsize=18)

for i in range(0,Np):
    ''' Sample zo from q(z)'''
    z0 = np.random.normal(mean1,cov1)
    ''' Scale the probability corresponding to z0'''
    u0 = k*norm(mean1,cov1).pdf(z0)*np.random.uniform()

    ''' Rejection based on p(z) comparison '''
    if( u0 < norm(mean2, cov2).pdf(z0)):
        plt.plot(z0,0,'ko',mfc='w',alpha=0.2,markersize=8)

plt.legend(frameon=False,fontsize=18)
plt.title(str(Np)+' samples',fontsize=16)

plt.savefig(img_dir+'Rejection_sampling.png')
# -


