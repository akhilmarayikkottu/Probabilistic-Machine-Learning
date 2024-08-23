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

# # Likelihood, Priors and Posteriors

# ## General idea of Beyesian inference

#  Given a set of parameters $\vec{x}$ and corresponding targets $\vec{t}$ combined as data $\mathcal{D} = \{\vec{x}, \vec{t} \}$, likelihood distribution $P(\mathcal{D}|\vec{p})$ gives the distribution of finding or obtaining the data given an argument or parameter $\vec{p}$ that defines the distribution function $P(\mathcal{D}| \vec{p})$. The argument or parameter $\vec{p}$ in the likelihood function can have a certain distribution which is defined as a prior knowledge distribution or prior distribution as $P(\vec{p})$. Here $\vec{p}$ can be a distribution as a function of some other argument $\vec{u}$. 
#  
#  Posterior distribution $P(\vec{p}|\mathcal{D})$ gives the probability of parameter $\vec{p}$ given the dataset $\mathcal{D}$. Posterior distribution is given using Beyes theorem as:
#  
#  $$\text{posterior } \propto \text{likelihood} \times \text{prior}$$
#  
#  $$ P(\vec{p}| \mathcal{D}) \propto P(\mathcal{D}|\vec{p}) \times P(\vec{p})$$
#  
#  By normalizing the equation
#  
#  $$ P(\vec{p}| \mathcal{D}) =  \frac {P(\mathcal{D}|\vec{p}) \times P(\vec{p})} {\int_\vec{p} P(\mathcal{D}|\vec{p}) \times P(\vec{p}) ~~d\vec{p} } $$
#  
#  
#  $$ P(\vec{p}| \mathcal{D}) =  \frac {P(\mathcal{D}|\vec{p}) \times P(\vec{p})} { P(\mathcal{D})   } $$
#  
#  

# ## Most popular parametric likelihood function : Normal distribution 

# $$\mathcal{N}(\vec{x}| \vec{\mu}, \Sigma) = \frac{1}{(2 \pi)^{1/N}} \frac{1}{\sqrt{|\Sigma|}} \exp \Bigg\{-\frac{(\vec{x}-\vec{\mu})^T \Sigma^{-1}(\vec{x}-\vec{\mu})}{2} \Bigg \}$$

# where $N$ is the dimension of $\vec{x} \in \mathrm{R}^N$. $\vec{\mu}$ is the vector of means and $\Sigma$ is the covariance matrix $\in \mathrm{R}^{N\times N}$. The inverse of $\Sigma$ is called the precision matrix $\Lambda = \Sigma^{-1} \in \mathrm{R}^{N \times N}$. $\Delta = (\vec{x}-\vec{\mu})^T \Sigma^{-1}(\vec{x}-\vec{\mu})$ is called the Mahalanohis distance.

# Consider two random variables $x_1$ and $x_2$, represented as $\vec{x} = (x_1, x_2)^T$ with mean $\vec{\mu} = (\mu_1, \mu_2)^T$ and covariance matrix $\Sigma =\begin{pmatrix} \sigma_{1,1} & \sigma_{1,2} \\ \sigma_{2,1}& \sigma_{2,2} \end{pmatrix}$, we can construct a joint probability distribution $\mathcal{N}(\vec{x}|\vec{\mu}, \Sigma)$. Note that the covariance matrix is symmetric as $\text{Cov}(x_1,x_2) = \int (x_1-\mu_1)(x_2-\mu_2) $ is an even function?

import numpy as np
from scipy import *
import matplotlib.pyplot as plt


def Normal(x,mu,sig):
    pdf= np.zeros(np.size(x))
    for i in range(0,np.size(x)):
        temp1  = 1/np.sqrt(2*np.pi)
        temp2  = 1/sig
        temp3  = (x[i]-mu)**2/(sig**2)
        pdf[i] = temp1*temp2*np.exp(-0.5*temp3)
    return(pdf)


def JointNormal(x1,x2,mu1,mu2,Sigma):
    pdf= np.zeros((np.size(x1),np.size(x2)))
    for i in range(0,np.size(x1)):
        for j in range (0,np.size(x2)):
            Det_sqrt = np.sqrt(np.linalg.det(Sigma))
            temp1    =  1/np.sqrt(2*np.pi)
            temp2    = np.array([[x1[i]-mu1],[x2[j]-mu2]])
            temp3    = np.dot(temp2.T,Sigma)
            temp4    = np.dot(temp3,temp2)
            pdf[j][i] =(temp1*np.exp(-0.5*temp4)/Det_sqrt)
    return(pdf)            


# +
x1 = np.linspace(-1,1,100)
x2 = np.linspace(-1,1,100)
mu1 = 0.0; mu2 = 0.0
Sigma = np.array([[1.5,0.5],[0.5,0.6]])

A = np.exp(-(x1-mu1)**2/2)/(2*np.pi*Sigma[0][0])**0.5
B = (np.exp(-(x2-mu2)**2/2)/(2*np.pi*Sigma[1][1])**0.5).reshape(-1,1)
C = A*B
# -

PDF = JointNormal(x1,x2,mu1,mu2,Sigma)

# +
fig,axs = plt.subplots(1,2,sharey=True,figsize=(8,4))
axs[0].contourf(x1,x2,C,levels=50,cmap='jet')
axs[1].contourf(x1,x2,PDF,levels = 50,cmap ='jet')
axs[0].set_ylabel('$x_2$')
axs[0].set_xlabel('$x_1$'); axs[1].set_xlabel('$x_1$')

axs[0].text(0.02, 0.99, '(a)',
        verticalalignment='top', horizontalalignment='left',
        transform=axs[0].transAxes, fontsize=14)
axs[1].text(0.02, 0.99, '(b)',
        verticalalignment='top', horizontalalignment='left',
        transform=axs[1].transAxes, fontsize=14)
# -

# Joint probability distribution $P(x_1,x_2)$ with $\{\mu_1, \mu_2\} = (0,0)$ and covariance matrix $ \begin{pmatrix}1.5&0.5\\0.5&0.6 \end{pmatrix}$
#
# Figure (a) shows the product of the two marginal distributions corresponding to the variables $x_1$ and $x_2$ given by $\mathcal{N}_1(x_1|\mu_1,\sigma_{1,1})$ and $\mathcal{N}_2(x_2|\mu_2,\sigma_{2,2})$ respectively. Figure (b) shows the "true" joint probability distribution of the two random variables. Note that $P(x_1,x_2) \neq P(x_1) \times P(x_2)$ as the covariance matrix is non-diagonal. That is, $\sigma_{i,j} \neq 0 ~~~; ~~\forall i \neq j $. The variables are correlated.
#
# The correlated normal distributions can be decomposed as uncorrelated distributions by defining new variables that are orthoginal to each other by diagonalizing the precision matrix $\Lambda = \Sigma^{-1}$ $\rightarrow$ $D$ 
#
# $$\Lambda = \Sigma^{-1} = U~ D~ U^T$$
#
# Therefore,
# $$ \Delta = (\vec{x}-\vec{\mu})^T ~\Sigma ~~(\vec{x}-\vec{\mu})$$
#
# $$ \Delta = (\vec{x}-\vec{\mu})^T \bigg\{ ~U ~D ~ U^T \bigg\}~(\vec{x}-\vec{\mu})$$
#
# $$ \Delta = \underbrace{(\vec{x}-\vec{\mu})^T  ~U}_{\vec{y}^T} ~~~~ D ~~~~ \underbrace{U^T ~(\vec{x}-\vec{\mu})}_{\vec{y}} $$
#
# The decoupled normal distribution is given as :
#
# $$ \mathcal{N}(\vec{x}| \vec{\mu}, \Sigma) = \frac{1}{(2 \pi)^{D/2}} \frac{1} {\lambda_1.\lambda_2.... \lambda_D} \exp \bigg\{ - \sum_i \frac{y_i^2}{2 \lambda_i} \bigg\}$$
#
# $$ \Sigma = \begin{pmatrix} 0.913 & -0.407 \\ 0.407 &  0.913 \end{pmatrix}^T  \begin{pmatrix}1.723 & \\ & 0.377 \end{pmatrix} \begin{pmatrix} 0.913 & -0.407 \\ 0.407 &  0.913 \end{pmatrix}$$
#
#
#
#
# $$ y_1 = 0.913 (x_1-\mu_1)+0.407(x_2-\mu_2)$$  and $$y_2 = -0.407(x_1-\mu_1)+0.913(x_2-\mu_2)$$
#
# Combining variables, $z_1 = 0.913 x_1+0.407 x_2$,     $z_2 = -0.404 x_1 + 0.903 x_2$,    $\nu_1 = 0.913 \mu_1+ 0.407 \mu_2$,   and $\nu_2 = -0.407 \mu_1 + 0.913 \mu_2$

# +
x1 = np.linspace(-1,1,100)
x2 = np.linspace(-2,2,100)
mu1 = 0.0; mu2 = 0.0
Sigma = np.array([[1.723,0.0],[0.0,0.377]])

z1 = 0.913*x1+0.407*x2
z2 = -0.407*x1+0.913*x2
nu1 = 0.913*mu1+0.407*mu2
nu2 = -0.407*mu1+0.913*mu2

PDF2 = JointNormal(z1,z2,nu1,nu2,Sigma)

# +
fig,axs = plt.subplots(1,2,sharey=False,figsize=(8,4))
axs[0].contourf(z1,z2,PDF2,levels=50,cmap='jet')
axs[1].plot(z1, Normal(z1,nu1,1.723),'r',label='$z_1$')
axs[1].plot(z1, Normal(z1,nu1,0.377),'b',label='$z_2$')
axs[0].set_ylabel('$z_2 = -0.404 x_1 + 0.903 x_2$'); axs[1].set_ylabel('$p$')
axs[0].set_xlabel('$z_1 = 0.913 x_1+0.407 x_2$'); axs[1].set_xlabel('$z_1, z_2$')
axs[1].set_xlim(-1,1)
axs[1].legend(frameon=False)

axs[0].set_xlim(-1,1); axs[0].set_ylim(-1,1)

axs[0].text(0.02, 0.99, '(a)',
        verticalalignment='top', horizontalalignment='left',
        transform=axs[0].transAxes, fontsize=14)
axs[1].text(0.02, 0.99, '(b)',
        verticalalignment='top', horizontalalignment='left',
        transform=axs[1].transAxes, fontsize=14)
# -

# Figure (a) shows the joint probability distribution of the new variables $z_1$ and $z_2$, $P (z_1,z_2)$ and the corresponding marginal probability distributions are given in figure (b). The joint probability $P(z_1, z_2)$ can be obtained from the individual marginal probabilities that are uncorrelated :
#
# $$ P(z_1,z_2) = P(z_1) \times P(z_2)$$
#
# where, $P(z_1)$ and $P(z_2)$ are marginal probability distributions of $z_1$ and $z_2$ respectively.

# ## Priors for Normal distribution

# Normal distributions have two arguments or parameters mean $\vec{\mu}$ and covriance $\Sigma$ or variance $\sigma^2$. Therefore, based on the parameter that is varied, we can have two priors corresponding to mean and variance

# ### Prior for mean

# For a univariant Normal distribution $\mathcal{N}(x| \mu, \sigma^2)$:
#
# $$\mathcal{N}(x| \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\bigg\{ - \frac{(x-\mu)^2}{2 \sigma^2}\bigg\} $$, prior for the mean should have a form:
#
# $$ $$



# ### Prior for variance or precision

#
#
# Gamma function has the form: 
# $$Gam(\lambda|a,b) = \frac{1}{\Gamma(a)} b^a \lambda^{a-1} \exp \Bigg(-b \lambda \Bigg)$$

def Gam(a,b,lamb):
    t1 = special.gamma(a)
    t2 = b**a*lamb**(a-1)*np.exp(-1*b*lamb)
    return(t2/t1)


def Norm(mu,sigma,x):
    t1 = (np.sqrt(2*np.pi*sigma**2))**(-1)
    t2 = np.exp(-0.5*(x-mu)**2/sigma**2)
    return(t1*t2)


# +
lamb = np.linspace(0.0001,2,1000)

plt.plot(lamb,Gam(0.01,0.01,lamb))
plt.plot(lamb,Gam(.1,.1,lamb))
plt.plot(lamb,Gam(1,1,lamb))
plt.plot(lamb,Gam(2,2,lamb))
plt.plot(lamb,Gam(3,3,lamb))
plt.xlim(0,2); plt.ylim(0,2)

# +
x = np.linspace(-2,2,2000)

plt.plot(x,Norm(0,0.1,x))
plt.plot(x,Norm(0,0.2,x))
plt.plot(x,Norm(0,0.3,x))
plt.xlim(-2,2); plt.ylim(0,4.25)

# +
sigma = np.linspace(0.001, 2,1000)
sigmainv = sigma.reshape(-1,1)
mu = np.linspace(-2,2,2000)
h = Norm(0,1,mu)*Gam(2,3,sigmainv)

plt.figure(figsize=(5,5))
plt.contourf(mu,sigma,h,cmap='jet',levels=100)
plt.xlabel('$\mu$'); plt.ylabel('$\sigma$')


# +
lamb = np.linspace(0.0001,10,1000)
for i in range (50,55):
    a0 = 1; b0 = 1
    a_n = a0+0.5*i
    b_n = b0+0.5*i*0.5
    plt.plot(lamb,Gam(a_n,b_n,lamb))
    
    
    
# -


