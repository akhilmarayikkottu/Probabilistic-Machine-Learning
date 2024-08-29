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
import functions
import random

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# Suppose we have multiple polynimial models order 1 to 5. We assume that a Gaussian noisy data is generated from the cubic model $( y = 2+3x+6x^2+5x^3)$ for $x \in [x_{lo},x_{hi}]$ with a gaussian noise of standard deviation $\beta$ and mean zero as shown below

beta = 0.5
x_lo = -1; x_hi = 1
Np   = 100
alpha = 1

# +
x_cont = np.linspace(-1,1,100)

_Poly = functions.poly
true_generator = _Poly.Cubic(x_cont,2,3,6,5)


noisy_data = []; x_obs=[]
for i in range (0,Np):
    x = random.uniform(x_lo,x_hi)
    x_obs.append(x)
    y = _Poly.Cubic(x,2,3,6,5)+random.gauss(0,beta)
    noisy_data.append(y)

noisy_data = np.array(noisy_data)

plt.plot(x_obs, noisy_data, 'bo', mfc='w',label = 'Generated data')
plt.plot(x_cont,true_generator,'r', label='True model')

plt.xlim(x_lo,x_hi)
plt.legend(fontsize=16,frameon=False)
plt.xlabel('$x$',fontsize=18)
plt.ylabel('$y$',fontsize=18)

plt.savefig('Images/Evidence_1.png')
# -

# For a constant prior distribution standard deviation $\alpha$, we can try to evaluate the maximum likelihood estimate, which is the maximum posterior estimate for each of these polynomial models of varying complexity or order. Hessian matrix or covariance matrix for each model is computed as $A_{order}$. For maximum likelihood estimates Hessian and Covariance are similar in the MLE vicinity.
#
# $$A = \alpha \mathcal{I} + \beta \Phi^T \Phi  $$
#
# The weight vector that maximizes the negative log likelihood indicated as $w_{ML,order}$
#
# $$ w_{ML,order} = \beta A^{-1} \Phi^T \bar{t}$$
#

Phi = functions.Basis

A_1 = alpha*np.identity(2)+beta*np.matmul(Phi.Linear(Np,x_obs).T, Phi.Linear(Np,x_obs))
A_2 = alpha*np.identity(3)+beta*np.matmul(Phi.Quadratic(Np,x_obs).T, Phi.Quadratic(Np,x_obs))
A_3 = alpha*np.identity(4)+beta*np.matmul(Phi.Cubic(Np,x_obs).T, Phi.Cubic(Np,x_obs))
A_4 = alpha*np.identity(5)+beta*np.matmul(Phi.Forth(Np,x_obs).T, Phi.Forth(Np,x_obs))
A_5 = alpha*np.identity(6)+beta*np.matmul(Phi.Fifth(Np,x_obs).T, Phi.Fifth(Np,x_obs))


w_ML_1 = beta*np.matmul(np.linalg.inv(A_1),np.matmul(Phi.Linear(Np,x_obs).T,noisy_data))
w_ML_2 = beta*np.matmul(np.linalg.inv(A_2),np.matmul(Phi.Quadratic(Np,x_obs).T,noisy_data))
w_ML_3 = beta*np.matmul(np.linalg.inv(A_3),np.matmul(Phi.Cubic(Np,x_obs).T,noisy_data))
w_ML_4 = beta*np.matmul(np.linalg.inv(A_4),np.matmul(Phi.Forth(Np,x_obs).T,noisy_data))
w_ML_5 = beta*np.matmul(np.linalg.inv(A_5),np.matmul(Phi.Fifth(Np,x_obs).T,noisy_data))


# +
plt.plot(x_obs, noisy_data, 'bo', mfc='w',alpha=0.2)
plt.plot(x_cont,true_generator,'r')
plt.plot(x_cont,_Poly.Linear(x_cont,w_ML_1[0],w_ML_1[1]),'k',linewidth=0.5)
plt.plot(x_cont,_Poly.Quadratic(x_cont,w_ML_2[0],w_ML_2[1],w_ML_2[2]),'k',linewidth=0.5)
plt.plot(x_cont,_Poly.Cubic(x_cont,w_ML_3[0],w_ML_3[1],w_ML_3[2],w_ML_3[3]),'g',linewidth=2)
plt.plot(x_cont,_Poly.Forth(x_cont,w_ML_4[0],w_ML_4[1],w_ML_4[2],w_ML_4[3],w_ML_4[4]),'k',linewidth=0.5)
plt.plot(x_cont,_Poly.Fifth(x_cont,w_ML_5[0],w_ML_5[1],w_ML_5[2],w_ML_5[3],w_ML_5[4],w_ML_5[5]),'k',linewidth=0.5)
plt.xlim(x_lo,x_hi)

plt.xlabel('$x$',fontsize=18)
plt.ylabel('$y$',fontsize=18)


plt.savefig('Images/Evidence_2.png')
# -

# When the error function is expanded quadratically in the vicinity of the MAP estimate in terms of the weight:
#
# $$ E(\bar{w})= E(\bar{w}_{ML})+ \frac{1}{2} (\bar{w}-\bar{w}_{ML})^T A  (\bar{w}-\bar{w}_{ML})  $$
#
# Evidence function can be approximated as weight marginalized likelihood function. After the quadratic expansion of the likelihood function around MAP, and integrating over weight parameter:
#
# $$ p(\mathcal{D} | \alpha, \beta) = \bigg( \frac{\beta}{2 \pi} \bigg)^{N/2}\bigg( \frac{\alpha}{2 \pi} \bigg)^{M/2} exp\bigg(-E(\bar{w}_{ML}) \bigg) \frac{(2 \pi)^{M/2}} {|A|^{1/2}} $$
#
# Logorithm of evidence is:
#
# $$ \text{log} \Bigg[p(\mathcal{D} | \alpha, \beta) \Bigg] = \frac{M}{2} \text{log} \alpha +\frac{N}{2} \text{log} \beta - E(\bar{w}_{ML}) - \frac{ \text{log} |A|} {2} - \frac{N}{2} \text{log} (2 \pi) $$

# +
Evidence = np.zeros(5)

Evidence[0] = functions.Evidence(noisy_data,Phi.Linear(Np,x_obs),w_ML_1,beta,alpha,2,Np,A_1)
Evidence[1] = functions.Evidence(noisy_data,Phi.Quadratic(Np,x_obs),w_ML_2,beta,alpha,3,Np,A_2)
Evidence[2] = functions.Evidence(noisy_data,Phi.Cubic(Np,x_obs),w_ML_3,beta,alpha,4,Np,A_3)
Evidence[3] = functions.Evidence(noisy_data,Phi.Forth(Np,x_obs),w_ML_4,beta,alpha,5,Np,A_4)
Evidence[4] = functions.Evidence(noisy_data,Phi.Fifth(Np,x_obs),w_ML_5,beta,alpha,6,Np,A_5)


# +
x_axis = np.linspace(2,6,5)

plt.plot(x_axis,Evidence,marker='x',color='b')
plt.xlim(2,6)

plt.xlabel('$M$',fontsize=18)
plt.ylabel('log $  p(\mathcal{D}  )$',fontsize=18)


plt.savefig('Images/Evidence_3.png')
