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

# # Bayesian Linear regression

# Consider a model (Linear regression model) of the form :
#     
#   $$y = w_0 + w_1. x$$
#   
#   
#   Here,  $\vec{w}$ = {$w_0, w_1$} is the parametric space for the model, $x$ is the input argument and $y$ is the target or the output. Assume that the parameters are close to $\sim [-1,1]$ range and independent of each other and both with a precision of $\alpha$ = 2.0 and mean around 0. Therefore, the joint probability distribution of the parameters $\vec{w}$ or thre prior knowledge $P(\vec{w})$ is the product of their individual marginal distirbutions:
#   
#   $$P (\vec{w}) = P(w_0,w_1) = P(w_0). P(w_1)$$
#   
#   Now assume that bothe these parameters follow a Gaussian:
#   
#   $$P (\vec{w}) = \mathcal{N}(w_0|0,\alpha).\mathcal{N}(w_1|0,\alpha)$$

#import FUN 
import random
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
import functions
import matplotlib as mpl
from matplotlib import  ticker


def model(w0,w1,x):
    y = w0+w1*x
    return(y)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# +
x, y = np.mgrid[-1:1:.001, -1:1:.001]
pos = np.dstack((x, y))

xsam = np.linspace(-1,1,10)

beta       = 0.2
alpha      = 0.2
Sigma_data = beta*np.array([[1,0],[0,1]])
# -

# ## Initially when we have only the prior distribution

# +
prior_mean = [0.0,0.0]
prior_covariance = [[alpha, 0], [0, alpha]]
prior = multivariate_normal(prior_mean, prior_covariance )

fig,axs = plt.subplots(1,3,sharey=False,figsize=(22,6))
axs[0].set_xlabel('$w_0$',fontsize=22); axs[0].set_ylabel('$w_1$',fontsize=22)
axs[1].set_xlabel('$w_0$',fontsize=22); axs[1].set_ylabel('$w_1$',fontsize=22)
axs[2].set_xlabel('$x$',fontsize=22); axs[2].set_ylabel('$y$',fontsize=22)
axs[0].set_xlim(-1,1);axs[0].set_ylim(-1,1)
axs[1].set_xlim(-1,1);axs[1].set_ylim(-1,1)
axs[2].set_xlim(-1,1);axs[2].set_ylim(-1,1)

axs[1].contourf(x, y, prior.pdf(pos),cmap='jet',levels=100)

for i in range (0,100):
    w0 = random.uniform(-1,1)
    w1 = random.uniform(-1,1)
    W  = [w0,w1]
    prob = prior.pdf(W)
    if(prob > random.random()):
        axs[2].plot(xsam,model(w0,w1,xsam),'b',linewidth=0.5)

axs[2].plot(xsam,model(-0.3,0.5,xsam),'r',linewidth=2)
axs[0].tick_params(axis='both', which='major', labelsize=18)
axs[1].tick_params(axis='both', which='major', labelsize=18)
axs[2].tick_params(axis='both', which='major', labelsize=18)

axs[0].set_title('likelihood',fontsize=24)
axs[1].set_title('prior/posterior',fontsize=24)
axs[2].set_title('data space',fontsize=24)

axs[0].plot(-0.3,0.5, 'wx', markersize=22)
axs[1].plot(-0.3,0.5, 'wx', markersize=22)

plt.savefig('Images/Stage0.png')
# -

# ### Get an observed point varying with a Gaussian of std dev 0.2

# +
x_sam1 = random.uniform(-1,1)
y_obs1 = model(-0.3,0.5,x_sam1)+random.gauss(0,beta)

w0 = np.linspace(-1,1,100)
w1 = w0

likelihood1 = np.zeros((np.size(w0),np.size(w1)))
for i in range(0,np.size(w0)):
    for j in range (0,np.size(w1)):
        Det_sqrt = np.sqrt(np.linalg.det(Sigma_data))
        temp1    =  1/np.sqrt(2*np.pi)
        temp2    = -0.5*((w0[i]+w1[j]*x_sam1))**2/beta**2
        likelihood1[i][j] = temp1*np.exp(temp2)/Det_sqrt

# +
Phi = functions.BayLinReg(1,[x_sam1])
SN_inv = alpha*np.identity(2)+beta*np.matmul(Phi.T,Phi)

mN = np.matmul(beta*np.linalg.inv(SN_inv),np.matmul(Phi.T, [y_obs1]))

posterior1 = multivariate_normal(mN, np.linalg.inv(SN_inv) )

# +
fig,axs = plt.subplots(1,3,sharey=False,figsize=(22,6))
axs[0].set_xlabel('$w_0$',fontsize=22); axs[0].set_ylabel('$w_1$',fontsize=22)
axs[1].set_xlabel('$w_0$',fontsize=22); axs[1].set_ylabel('$w_1$',fontsize=22)
axs[2].set_xlabel('$x$',fontsize=22); axs[2].set_ylabel('$y$',fontsize=22)
axs[0].set_xlim(-1,1);axs[0].set_ylim(-1,1)
axs[1].set_xlim(-1,1);axs[1].set_ylim(-1,1)
axs[2].set_xlim(-1,1);axs[2].set_ylim(-1,1)
axs[2].plot(xsam,model(-0.3,0.5,xsam),'r',linewidth=2)
axs[0].tick_params(axis='both', which='major', labelsize=18)
axs[1].tick_params(axis='both', which='major', labelsize=18)
axs[2].tick_params(axis='both', which='major', labelsize=18)
axs[2].plot(x_sam1,y_obs1,'ko',markersize=18,mfc='w')


axs[0].contourf(w0,w1,likelihood1,cmap='jet',levels=250)

axs[1].contourf(x, y, posterior1.pdf(pos),cmap='jet',levels=250)

for i in range (0,100):
    w0 = random.uniform(-1,1)
    w1 = random.uniform(-1,1)
    W  = [w0,w1]
    prob = posterior1 .pdf(W)
    if(prob > random.uniform(0,1)):
        axs[2].plot(xsam,model(w0,w1,xsam),'b',linewidth=0.5)

axs[0].set_title('likelihood',fontsize=24)
axs[1].set_title('prior/posterior',fontsize=24)
axs[2].set_title('data space',fontsize=24)

axs[0].plot(-0.3,0.5, 'wx', markersize=22)
axs[1].plot(-0.3,0.5, 'wx', markersize=22)

plt.savefig('Images/Stage1.png')
# -

# ## Observe second datapoint 

# +
x_sam2 = random.uniform(-1,1)
y_obs2 = model(-0.3,0.5,x_sam2)+random.gauss(0,beta)

w0 = np.linspace(-1,1,100)
w1 = w0
likelihood2 = np.zeros((np.size(w0),np.size(w1)))
for i in range(0,np.size(w0)):
    for j in range (0,np.size(w1)):
        Det_sqrt = np.sqrt(np.linalg.det(Sigma_data))
        temp1    =  1/np.sqrt(2*np.pi)
        temp2    = -0.5*((w0[i]+w1[j]*x_sam2))**2/beta**2
        likelihood2[i][j] = temp1*np.exp(temp2)/Det_sqrt

# +
Phi = functions.BayLinReg(2,[x_sam1,x_sam2])
SN_inv = alpha*np.identity(2)+beta*np.matmul(Phi.T,Phi)

mN = np.matmul(beta*np.linalg.inv(SN_inv),np.matmul(Phi.T, [y_obs1,y_obs2]))

posterior2 = multivariate_normal(mN, np.linalg.inv(SN_inv) )

# +
fig,axs = plt.subplots(1,3,sharey=False,figsize=(22,6))
axs[0].set_xlabel('$w_0$',fontsize=22); axs[0].set_ylabel('$w_1$',fontsize=22)
axs[1].set_xlabel('$w_0$',fontsize=22); axs[1].set_ylabel('$w_1$',fontsize=22)
axs[2].set_xlabel('$x$',fontsize=22); axs[2].set_ylabel('$y$',fontsize=22)
axs[0].set_xlim(-1,1);axs[0].set_ylim(-1,1)
axs[1].set_xlim(-1,1);axs[1].set_ylim(-1,1)
axs[2].set_xlim(-1,1);axs[2].set_ylim(-1,1)
axs[2].plot(xsam,model(-0.3,0.5,xsam),'r',linewidth=2)
axs[0].tick_params(axis='both', which='major', labelsize=18)
axs[1].tick_params(axis='both', which='major', labelsize=18)
axs[2].tick_params(axis='both', which='major', labelsize=18)
axs[2].plot(x_sam1,y_obs1,'ko',markersize=18,mfc='w')
axs[2].plot(x_sam2,y_obs2,'ko',markersize=18,mfc='w')


axs[0].contourf(w0,w1,likelihood2,cmap='jet',levels=50)

axs[1].contourf(x, y, posterior2.pdf(pos),cmap='jet', levels=250)

for i in range (0,100):
    w0 = random.uniform(-1,1)
    w1 = random.uniform(-1,1)
    W  = [w0,w1]
    prob = posterior2 .pdf(W)
    if(prob > random.uniform(0,1)):
        axs[2].plot(xsam,model(w0,w1,xsam),'b',linewidth=0.5)

axs[0].set_title('likelihood',fontsize=24)
axs[1].set_title('prior/posterior',fontsize=24)
axs[2].set_title('dataspace',fontsize=24)

axs[0].plot(-0.3,0.5, 'wx', markersize=22)
axs[1].plot(-0.3,0.5, 'wx', markersize=22)

plt.savefig('Images/Stage2.png')
# -

# ## Generating $N$ observed points

# +
Np = 100

x_obs = np.random.uniform(-1,1,size=Np)

y_obs = np.zeros(Np)
for i in range (0,Np):
    y_obs[i] = model(-0.3,0.5,x_obs[i])+random.gauss(0,beta)

w0 = np.linspace(-1,1,100)
w1 = w0
likelihood3 = np.zeros((np.size(w0),np.size(w1)))
for i in range(0,np.size(w0)):
    for j in range (0,np.size(w1)):
        Det_sqrt = np.sqrt(np.linalg.det(Sigma_data))
        temp1    =  1/np.sqrt(2*np.pi)
        temp2    = -0.5*((w0[i]+w1[j]*x_obs[-1]))**2/beta**2
        likelihood3[i][j] = temp1*np.exp(temp2)/Det_sqrt

# +
Phi = functions.BayLinReg(Np,x_obs)
SN_inv = alpha*np.identity(2)+beta*np.matmul(Phi.T,Phi)

mN = np.matmul(beta*np.linalg.inv(SN_inv),np.matmul(Phi.T, y_obs))

posterior3 = multivariate_normal(mN, np.linalg.inv(SN_inv) )

# +
fig,axs = plt.subplots(1,3,sharey=False,figsize=(22,6))
axs[0].set_xlabel('$w_0$',fontsize=22); axs[0].set_ylabel('$w_1$',fontsize=22)
axs[1].set_xlabel('$w_0$',fontsize=22); axs[1].set_ylabel('$w_1$',fontsize=22)
axs[2].set_xlabel('$x$',fontsize=22); axs[2].set_ylabel('$y$',fontsize=22)
axs[0].set_xlim(-1,1);axs[0].set_ylim(-1,1)
axs[1].set_xlim(-1,1);axs[1].set_ylim(-1,1)
axs[2].set_xlim(-1,1);axs[2].set_ylim(-1,1)
axs[2].plot(xsam,model(-0.3,0.5,xsam),'r',linewidth=2)
axs[0].tick_params(axis='both', which='major', labelsize=18)
axs[1].tick_params(axis='both', which='major', labelsize=18)
axs[2].tick_params(axis='both', which='major', labelsize=18)
axs[2].plot(x_obs,y_obs,'ko',markersize=18,mfc='w')

axs[0].contourf(w0,w1,likelihood3,cmap='jet',levels=50)

axs[1].contourf(x, y, posterior3.pdf(pos),cmap='jet', levels=250)

for i in range (0,100):
    w0 = random.uniform(-1,1)
    w1 = random.uniform(-1,1)
    W  = [w0,w1]
    prob = posterior3 .pdf(W)
    if(prob > random.uniform(0,1)):
        axs[2].plot(xsam,model(w0,w1,xsam),'b',linewidth=0.5)

axs[0].set_title('likelihood',fontsize=24)
axs[1].set_title('prior/posterior',fontsize=24)
axs[2].set_title('dataspace',fontsize=24)

axs[0].plot(-0.3,0.5, 'wx', markersize=22)
axs[1].plot(-0.3,0.5, 'wx', markersize=22)


plt.savefig('Images/Stage3.png')
# -


