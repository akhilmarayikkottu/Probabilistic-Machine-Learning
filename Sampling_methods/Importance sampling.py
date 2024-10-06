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
mean1 = [0.0]
cov1  = [2.0]

mean2 = [-1.0]
cov2  = [1.0] 

k     = 2.5 
Np    = [1,2,5,20,200,2000,20000]

rv1   = norm(mean1, cov1)
rv2   = norm(mean2, cov2)

x_cont = np.linspace(-5,5,1000)

z_min = -5
z_max =  5


# +
def f(z):
    return(3+0.005*z**3)

def q(z):
    return(k*rv1.pdf(z))

def p_by_q(z):
    p = rv2.pdf(z)
    q = k*rv1.pdf(z)
    return(p/q )

def f_p(z):
    t = rv2.pdf(z)*f(z)
    return(t)


# +
fig, ax1 = plt.subplots(figsize=(6,4))
ax2 = ax1.twinx()

ax1.set_xlabel('$z$',fontsize=18)
ax1.set_ylabel('$ p $', fontsize=18)
ax2.set_ylabel('$f(z)$', fontsize=18)
ax1.set_xlim(-5,5);ax2.set_ylim(2.4,3.6)

ax1.plot(x_cont, k*rv1.pdf(x_cont),'r')
ax1.fill_between(x_cont, rv2.pdf(x_cont),k*rv1.pdf(x_cont),color='gray',alpha=0.3)
ax1.plot(x_cont, rv2.pdf(x_cont),'b')
ax2.plot(x_cont, f(x_cont), 'k')

plt.savefig(img_dir+'Importance_sampling_config.png')
# -

_sum_list = []
for j in range (0,7):
    N = Np[j]
    sum = 0.0
    for i in range (0,N):
        dz   = (z_max-z_min)/N
        z_lo = z_min+i*dz
        z_hi = z_lo+dz
        z_c  = z_lo+0.5*dz
        sum  = sum+f(z_c)*p_by_q(z_c)*q(z_c)*dz
    _sum_list.append(sum)

# +
plt.figure(figsize=(6,4))
plt.plot(Np,2.9799*np.ones(7),'r--',label='True expectation')
plt.semilogx(Np,_sum_list, 'b', marker='o',markersize=10,mfc='w', label='Importance sampling')

plt.xlabel('Number of elements',fontsize=16)
plt.ylabel('$\mathrm{E}(f(z))$',fontsize=20)
plt.legend(fontsize=16,frameon=False)

 
plt.savefig(img_dir+'Importance_sampling_expectation.png')
