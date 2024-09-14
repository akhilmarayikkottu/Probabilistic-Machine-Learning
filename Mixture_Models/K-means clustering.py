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
Np_1 = 300
Np_2 = 500
N_p  = Np_1+Np_2

mean_1 = [-5,-5]
mean_2 = [5, 5]

cov_1 = [[2,.2],[.2,3]]
cov_2 = [[5,.1],[.1,3]]

N_k   = 2
mu_k  = np.array([[5,-5],[-5,0]])

rnk   = np.zeros((N_p,N_k))

n_iters = 10
plot_pnts = 1

# +
C1  = np.random.multivariate_normal(mean_1, cov_1, Np_1)
C2  = np.random.multivariate_normal(mean_2, cov_2, Np_2)

X   = np.concatenate((C1,C2))
# -

plt.figure(figsize=(6,6))
plt.scatter(X[:,0],X[:,1], marker = 'x', color='k')
plt.plot(mu_k[0][0],mu_k[0][1],'ro',mfc='w', markersize=14)
plt.plot(mu_k[1][0],mu_k[1][1],'bo',mfc='w', markersize=14)
plt.xlim(-12,12); plt.ylim(-12,12)
plt.savefig(img_dir+'InitialDataSet.png')

# ## Assign datapoints to cluster centers

loss = []
for iter in range (0,n_iters):

    _total_distance = 0.0
    for i in range (0,N_p):
        data = X[i,:]
        distance = np.zeros(N_k)
        for k in range (0,N_k):
            distance[k] = Modules.dist(data,mu_k[k,:])
        #print(distComp(distance),distance)
        index,distance = Modules.distComp(distance)
        _total_distance +=  distance

        rnk[i,index] = 1
    loss.append(_total_distance)

    mu_k  = np.zeros((N_k,2))
    sum_r = np.zeros(N_k)
    for i in range (0,N_p):
        data = X[i,:]
        r    = rnk[i,:]
        sum_r+=r
        for k in range (0,N_k):
            mu_k[k,:]+= data*r[k]
    
    for k in range (0,N_k):
        mu_k[k,:] = mu_k[k,:]/sum_r[k]

    if (iter%plot_pnts == 0):
        Modules.plotCluster(X,rnk,800,mu_k,img_dir,iter)

plt.figure(figsize=(8,5))
plt.plot(loss,'b',marker='o',mfc='w',mec='r',markersize=10)
plt.xlabel('Iterations',fontsize=18)
plt.ylabel('$J$',fontsize=22)
plt.savefig(img_dir+'KMeansLoss.png')


