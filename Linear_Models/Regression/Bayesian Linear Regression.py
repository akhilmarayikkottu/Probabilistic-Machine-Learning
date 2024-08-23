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

import FUN 
import random
import numpy as np
from scipy import *
import matplotlib.pyplot as plt


def JointNormalSamp(x1,x2,mu1,mu2,Sigma):
    import numpy as np
    Det_sqrt = np.sqrt(np.linalg.det(Sigma))
    temp1    =  1/np.sqrt(2*np.pi)
    temp2    = np.array([[x1-mu1],[x2-mu2]])
    temp3    = np.dot(temp2.T,Sigma)
    temp4    = np.dot(temp3,temp2)
    pdf      =(temp1*np.exp(-0.5*temp4)/Det_sqrt)
    return(pdf)  


def model(w0,w1,x):
    y = w0+w1*x
    return(y)


# +
w0 = np.linspace(-1,1,100)
w1 = w0
mu0 = 0
mu1 = 0

Sigma = np.array([[0.2,0],[0,0.2]])
PRIOR = FUN.JointNormal(w0,w1,mu0,mu1,Sigma)

# +
fig,axs = plt.subplots(1,2,sharey=False,figsize=(9,4))
axs[0].contourf(w0,w1,PRIOR,cmap='jet',levels=250)
axs[0].set_xlabel('$w_0$'); axs[0].set_ylabel('$w_1$')

Nsam = 10
xsam = np.linspace(-1,1,10)
for i in range(0,Nsam):
    w0sam = -1+2*random.random()
    w1sam = -1+2*random.random()
    Prob  =JointNormalSamp(w0sam,w1sam,mu0,mu1,Sigma)
    # acceprance rejection
    if(Prob > random.random()):
        axs[1].plot(xsam,model(w0sam,w1sam,xsam),'k')

axs[1].plot(xsam,model(-0.3,0.5,xsam),'r',linewidth=4)
axs[1].set_xlabel('$x$');axs[1].set_ylabel('$y$')    
axs[1].set_xlim(-1,1);axs[1].set_ylim(-1,1)

# -

# Figure one shows the prior distribution $P(w_0, w_1)$ and figure two shows the realization of the model $y = w_0+w_1.x$ for randomly picked parameter points $\vec{w}^*$ from the prior distribution. Note that the 

# Assume that the true data points are from a line $t = -0.3+0.5x$ with a noise level of $\beta = 25$. For example, at $x = 0.5$, the true value is -0.05. Adding a bit of noise to it $\sim \mathcal{N}(0,0.02)$. The new point with the noise can be something like 
#
# $$\tilde{t} = -0.05 + \mathcal{N}(0, 0.02) \sim -0.0343$$
#
# The distribution of the likelihood function which shows the distribution of the predicted target $\tilde{t}$ for a $\tilde{x}$, $P(\tilde{t}|\vec{w})$ is given as:
#
# $$P(\tilde{t}| \tilde{x}, \vec{w})   = \mathcal{N}\bigg(\tilde{t}\bigg| ~\underbrace{w_0+w_1 \tilde{x}}_\text{model}~~, \beta \bigg) $$
#
# $$P(\tilde{t}|\tilde{x}, \vec{w})   \propto \exp \Bigg( \frac{[\tilde{t} - (w_0 + w_1. \tilde{x})]^T.[\tilde{t} - (w_0 + w_1. \tilde{x})]}{\beta} \Bigg) $$

# +
#-0.05+np.random.normal(0,0.02)
np.size(w0)
beta       = .2
Sigma_data = beta*np.array([[1,0],[0,1]])

t_tilde = -0.05+np.random.normal(0,beta)
x_tilde =  0.5
Likelihood = np.zeros((np.size(w0),np.size(w1)))
for i in range(0,np.size(w0)):
    for j in range (0,np.size(w1)):
        Det_sqrt = np.sqrt(np.linalg.det(Sigma_data))
        temp1    =  1/np.sqrt(2*np.pi)
        temp2    = -0.5*((w0[i]+w1[j]*x_tilde))**2/beta**2
        Likelihood[i][j] = temp1*np.exp(temp2)/Det_sqrt

# +
fig,axs = plt.subplots(1,2,sharey=False,figsize=(9,4))
axs[0].contourf(w0,w1,Likelihood.T,cmap='jet',levels=250)
axs[0].set_xlabel('$w_0$'); axs[0].set_ylabel('$w_1$')
axs[0].plot(-0.3,0.5,'wo',markersize=10,mfc='none')

Nsam = 10
xsam = np.linspace(-1,1,10)
for i in range(0,Nsam):
    w0sam = -1+2*random.random()
    w1sam = -1+2*random.random()
    Prob  =JointNormalSamp(w0sam,w1sam,mu0,mu1,Sigma)
    # acceprance rejection
    if(Prob > random.random()):
        axs[1].plot(xsam,model(w0sam,w1sam,xsam),'k')

axs[1].plot(xsam,model(-0.3,0.5,xsam),'r',linewidth=4)
axs[1].set_xlabel('$x$');axs[1].set_ylabel('$y$')    
axs[1].set_xlim(-1,1);axs[1].set_ylim(-1,1)
axs[1].plot(x_tilde,t_tilde,'bo',markersize=12,mfc='none')

# +
plt.figure(figsize=(4,4))
posterior = PRIOR*Likelihood.T

plt.contourf(w0,w1,posterior,cmap='jet',levels=250)
plt.plot(-0.3,0.5,'wo',markersize=10,mfc='none')

# +
x_tilde2 = -.25
t_tilde2 = model(-0.3,0.5,x_tilde2)+np.random.normal(0,beta)

Likelihood2 = np.zeros((np.size(w0),np.size(w1)))
for i in range(0,np.size(w0)):
    for j in range (0,np.size(w1)):
        Det_sqrt = np.sqrt(np.linalg.det(Sigma_data))
        temp1    =  1/np.sqrt(2*np.pi)
        temp2    = -0.5*(t_tilde2-model(w0[i],w1[j],x_tilde2))**2/beta**2
        Likelihood2[i][j] = temp1*np.exp(temp2)/Det_sqrt

# +
fig,axs = plt.subplots(1,2,sharey=False,figsize=(9,4))
axs[0].contourf(w0,w1,Likelihood2.T,cmap='jet',levels=250)
axs[0].set_xlabel('$w_0$'); axs[0].set_ylabel('$w_1$')
axs[0].plot(-0.3,0.5,'wo',markersize=10,mfc='none')

Nsam = 10
xsam = np.linspace(-1,1,10)
for i in range(0,Nsam):
    w0sam = -1+2*random.random()
    w1sam = -1+2*random.random()
    Prob  =JointNormalSamp(w0sam,w1sam,mu0,mu1,Sigma)
    # acceprance rejection
    if(Prob > random.random()):
        axs[1].plot(xsam,model(w0sam,w1sam,xsam),'k')

axs[1].plot(xsam,model(-0.3,0.5,xsam),'r',linewidth=4)
axs[1].set_xlabel('$x$');axs[1].set_ylabel('$y$')    
axs[1].set_xlim(-1,1);axs[1].set_ylim(-1,1)
axs[1].plot(x_tilde,t_tilde,'bo',markersize=12,mfc='none')
axs[1].plot(x_tilde2,t_tilde2,'bo',markersize=12,mfc='none')

# +
posterior = posterior*Likelihood2.T
plt.figure(figsize=(4,4))


plt.contourf(w0,w1,posterior,cmap='jet',levels=250)
plt.plot(-0.3,0.5,'wo',markersize=10,mfc='none')

# +
AA = Likelihood.T*Likelihood2.T
plt.figure(figsize=(4,4))


plt.contourf(w0,w1,AA,cmap='jet',levels=250)
plt.plot(-0.3,0.5,'wo',markersize=10,mfc='none')
# -

model


