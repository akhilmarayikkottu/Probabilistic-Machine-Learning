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
from scipy.stats import multivariate_normal
import random
import Modules

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

img_dir = 'Images/'

# +
N_k = 3

Np = np.array([1000, 1000,1000])
_tot_Np =  np.sum(Np)

mu_1 = [-5,-5]
mu_2 = [0, 0]
mu_3 = [5, -1.5]
mu   = np.array([mu_1,mu_2,mu_3])

cov_1 = [[2.0,-1.8],[-1.8,2.0]]
cov_2 = [[5,6],[6,10]]
cov_3 = [[3,-1.8],[-1.8,4]]
cov = np.array([cov_1,cov_2,cov_3])

pi  = np.ones(N_k)/N_k

Z_pstrior = np.zeros((_tot_Np, N_k))

# +
C1  = np.random.multivariate_normal(mu_1, cov_1, Np[0])
C2  = np.random.multivariate_normal(mu_2, cov_2, Np[1])
C3  = np.random.multivariate_normal(mu_3, cov_3, Np[2])

X   = np.concatenate((C1,C2,C3))

# +
plt.figure(figsize=(12,6))
plt.scatter(C1[:,0],C1[:,1], marker = 'x', color='k')
plt.scatter(C2[:,0],C2[:,1], marker = 'x', color='r')
plt.scatter(C3[:,0],C3[:,1], marker = 'x', color='b')

plt.xlim(-12,12); plt.ylim(-12,12)
plt.xlabel('$x_1$', fontsize=22)
plt.ylabel('$x_2$', fontsize=22)


plt.savefig(img_dir+'ThreeCluster.png')
# -

# ## 

# +
plt.figure(figsize=(12,6))
plt.scatter(X[:,0],X[:,1], marker = 'x', color='k')

plt.xlim(-12,12); plt.ylim(-12,12)
plt.xlabel('$x_1$', fontsize=22)
plt.ylabel('$x_2$', fontsize=22)
plt.savefig(img_dir+'IncompleteData.png')
# -

# Given that we have an observed data set which is incomplete (unknown mapping between $x_i$ and $z_i$), and observed parameters $\theta$ , we want to see the probability distribution of the latent variable $z_i$. That is, the most we can do with this information is to compute the posterior distribution of the latent variable $p(Z|X,\theta)$.
#
# From Bayes' theorem, posterior probability can be computed as,
#
# $$ p(Z|X,\theta) = \frac{p(Z) p(X|Z, \theta)}{\sum_z p(Z) p(X|Z, \theta)}$$
#
# If we assume that the mixture is generated from Gaussians, $\theta = \{ \mu, \Sigma, \pi \}$. The probabilities $p(Z)$ and $p(X|Z)$ can be (or should be) functions of the parameters $\theta$. In Gaussian mixture models, the latent variable is a $K$ dimensional binary vector corresponding to each datapoint $x_i$, $z_i = \{ z_1, z_2, . . . . z_k \}$. The vector can be in $K$ states corresponding to its assignment to each cluster (center). BUT, we can only give (approximate) probabilities for each of these states. The probability of each datapoint $x_i$ baing in one of the $K$ states in a Gaussian mixture model is given by the corresponding mixing ratio $\pi_k$ (simple !!!). For a given mixing ratio $z_k$, $p(X|Z, \theta)$ is a Gaussian.
#
# $$p(z_k) = \pi_k$$
#
# $$p(x_i|z_k,\theta) = \mathcal{N}(x_i| \mu_k, \Sigma_k)$$
#
#
# Therefore 
#
# $$p(z_k|x_i, \theta) = \frac{\pi_k \mathcal{N}(x_i|z_k,\Sigma_k)}{\sum_k \pi_k \mathcal{N}(x_i|z_k,\Sigma_k)}$$

for i in range (0, _tot_Np):
    data = X[i,:]
    z_hldr = np.zeros(N_k)
    for k in range (0,N_k):
        z_hldr[k] = pi[k]*multivariate_normal.pdf(data,mean= mu[k], cov=cov[k])
    _tot_z_hldr = np.sum(z_hldr)

    for k in range (0,N_k):
        Z_pstrior[i,k] = z_hldr[k]/_tot_z_hldr

# +
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 18), sharex=True, sharey=True)
ax[0].set_xlim(-12,12); ax[0].set_ylim(-12,12)

color = 'Reds'

ax[0].scatter(X[:,0],X[:,1], marker = 'x', c=Z_pstrior[:,0],cmap=color)
ax[1].scatter(X[:,0],X[:,1], marker = 'x', c=Z_pstrior[:,1],cmap=color)
ax[2].scatter(X[:,0],X[:,1], marker = 'x', c=Z_pstrior[:,2],cmap=color)

ax[2].set_xlabel('$x_1$', fontsize=22)
ax[0].set_ylabel('$x_2$', fontsize=22)
ax[1].set_ylabel('$x_2$', fontsize=22)
ax[2].set_ylabel('$x_2$', fontsize=22)


ax[0].set_title('$p(z_1|X,\mu, \Sigma)$',fontsize=22)
ax[1].set_title('$p(z_2|X,\mu, \Sigma)$',fontsize=22)
ax[2].set_title('$p(z_3|X,\mu, \Sigma)$',fontsize=22)


# -


