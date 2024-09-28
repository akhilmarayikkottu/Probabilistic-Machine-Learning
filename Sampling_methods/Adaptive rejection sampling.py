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
import Modules

# +
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

img_dir = 'Images/'

# +
mean2 = [-1]
cov2  = [1.0] 

k     = 2.5 
Np    = 2000
# -

rv2 = norm(mean2, cov2)

# +
x = np.linspace(-5,5,1000)
proposal_dist =[]
for i in range(0,1000):
    proposal_dist.append(Modules.envelope(x[i]))

z_c = [-3.5,0,3.5]
p_c = rv2.pdf(z_c)

plt.figure(figsize=(6,4))

plt.plot(z_c,p_c ,'ro',mfc='w',markersize=10)

plt.plot(x, proposal_dist,'r', label='envelope')
plt.fill_between(x,rv2.pdf(x),np.array(proposal_dist).reshape(-1),color='grey',alpha=0.5)

plt.semilogy(x,rv2.pdf(x),'b',label='$\\tilde{p}(z)$')

plt.xlim(-5,5);plt.ylim(0.0,1000)
plt.xlabel('$z$',fontsize=18)
plt.ylabel('log$\Bigg( p(z) \Bigg)$', fontsize=18)
plt.legend(frameon=False,fontsize=18)

plt.savefig(img_dir+'adaptive_reject_sampling_config.png')

# +
plt.figure(figsize=(6,4))

plt.plot(z_c,p_c ,'ro',mfc='w',markersize=10)
plt.plot(x,rv2.pdf(x),'b',label='$\\tilde{p}(z)$')
plt.xlim(-5,5);plt.ylim(-0.01,0.55)
plt.xlabel('$z$',fontsize=18)
plt.ylabel('log$\Bigg( p(z) \Bigg)$', fontsize=18)
plt.legend(frameon=False,fontsize=18)

for i in range (0,Np):
    _z0 = -5+np.random.uniform()*(10)
    _ez = Modules.envelope(_z0)

    u0  = np.random.uniform()*_ez
    if ( u0 < rv2.pdf(_z0)):
        plt.plot(_z0,0.0,'ko',mfc='w',alpha=.3)

plt.title(str(Np)+' samples',fontsize=16)
plt.savefig(img_dir+'adaptive_reject_sample.png')
