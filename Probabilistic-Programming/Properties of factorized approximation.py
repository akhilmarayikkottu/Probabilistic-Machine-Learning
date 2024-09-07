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
from matplotlib.animation import FuncAnimation, writers

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# Suppose iid samples of data is generated from a multivarient Normal distribution $\mathcal{N}(Z|\mu, \Lambda)$with correlated latent variables $Z = [z_1,z_2]$. The number of samples $N_p$ is a user-variable. Our aim is to approximate this ${\it true}$ distribution with a factorized distribution $q(Z) =  q_1(z_1)q_2(z_2)$. Using mean-field theory or factorized approximation, estimate for each latent variable can be estimated:
#
# $$ \text{ln} \bigg(q^*(z_1) \bigg) = \mathrm{E}_{z2} \Bigg[ \text{ln} \bigg( p(Z)\bigg) \Bigg]$$
#
# $$ \text{ln} \bigg(q^*(z_2) \bigg) = \mathrm{E}_{z1} \Bigg[ \text{ln} \bigg( p(Z)\bigg) \Bigg]$$
#
#
# By expanding the integral (keeping only the factors corresponding to the latent variable being optimized) and completing squares, we get the following Gaussians as the approximate distirbutions:
#
# $$q^*(z_1) = \mathcal{N} (z_1| m_1, \Lambda_{11}) $$
# $$q^*(z_2) = \mathcal{N} (z_2| m_2, \Lambda_{22}) $$
#
# where, $m_1 =  \mu_1-\Lambda_{11}^{-1} \Lambda_{12} (\mathrm{E}[z_2] - \mu_2) $ and  $m_2 =  \mu_2-\Lambda_{22}^{-1} \Lambda_{21} (\mathrm{E}[z_1] - \mu_1) $. Since $\mathrm{E}[z_i] = \mu_i$, $m_i = \mathrm{E}[z_i]$ (Makes life easy !!!).

# +
Np   = np.array([5,50,500,5000]) #Number of samples

mean = [0, 0]
cov  = [[2.0, -0.8], [-0.8, 0.5]] 
# -

rv = multivariate_normal(mean=None, cov=1, allow_singular=False)
rv = multivariate_normal(mean, cov)


# +
x, y = np.mgrid[-4:4:.01, -4:4:.01]
pos = np.dstack((x, y))

for i in range (0,4):
    Z  = np.random.multivariate_normal(mean, cov, int(Np[i]))
    z1 = Z[:,0]
    z2 = Z[:,1]
    plt.figure(figsize=(6,6))
    plt.contour(x, y, rv.pdf(pos),cmap='jet',levels=20,alpha=.6)
    plt.scatter(z1,z2, marker='x',s=15,color='k', alpha= 0.5)
    plt.scatter(0.1,-0.1, rv.pdf((.1,-.1)))
    plt.xlabel('$z_1$', fontsize=22)
    plt.ylabel('$z_2$', fontsize=22)
    plt.xlim(-4,4); plt.ylim(-4,4)
    plt.savefig('Images/multiv_1_'+str(i)+'.png')

    mean_approx = [np.mean(z1), np.mean(z2)]
    cov_approx  = [[np.cov(Z.T)[0][0],0],[0,np.cov(Z.T)[1][1]]]
    rv2 = multivariate_normal(mean_approx, cov_approx)
    plt.figure(figsize=(6,6))
    plt.contour(x, y, rv.pdf(pos),cmap='jet',levels=20,alpha=.3)
    plt.contour(x, y, rv2.pdf(pos),cmap='jet',levels=10,alpha=1)
    plt.xlabel('$z_1$', fontsize=22)
    plt.ylabel('$z_2$', fontsize=22)
    plt.savefig('Images/multiv_2_'+str(i)+'.png')
# -

# #### 
