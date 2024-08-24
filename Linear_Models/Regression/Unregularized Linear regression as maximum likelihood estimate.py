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
import random

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


# +
def true_output(x):
    return (10*x+2)
    
def noisy_obs(x_lo, x_hi, Np, sigma):
    x_list=[]
    t_list=[]
    for i in range (0,Np):
        x = random.uniform(x_lo,x_hi)
        t = true_output(x)+random.gauss(0,sigma)
        x_list.append(x)
        t_list.append(t)
    return(np.array(x_list), np.array(t_list))

def poly_model(a,b,c,d,e,f,x):
    return(a+b*x+c*x**2+d*x**3+e*x**4+f*x**5)

def ModelMatrix(Np,obs_x):
    N = np.ones((Np,6))
    for i in range (0,Np):
        N[i,1] = obs_x[i]
        N[i,2] = obs_x[i]**2
        N[i,3] = obs_x[i]**3
        N[i,4] = obs_x[i]**4
        N[i,5] = obs_x[i]**5
    return (N)


# +
x = np.linspace(-2,2,1000)

obs_x,obs_t  = noisy_obs(-2,2,100,10)

plt.plot(x,true_output(x),'b')
plt.plot(obs_x,obs_t, 'rx')
plt.xlim(-2,2)
plt.ylabel('$t$',fontsize=22); plt.xlabel('$x$',fontsize=22)
plt.savefig('Images/Raw_data.png')
plt.show()
# -

# ## Normal equation solution 

# +
A = ModelMatrix(np.size(obs_x),obs_x)
Phi = np.matmul(A.T,A)
Phi_rhs = np.matmul(A.T,obs_t)

Phi_inv = np.linalg.inv(Phi)

a,b,c,d,e,f = np.matmul(Phi_inv, Phi_rhs)

# +

plt.plot(x,true_output(x),'b')
plt.plot(obs_x,obs_t, 'rx')
plt.plot(x,poly_model(a,b,c,d,e,f,x),'g')
plt.xlim(-2,2)
plt.ylabel('$t$',fontsize=22); plt.xlabel('$x$',fontsize=22)
plt.savefig('Images/MLE_out.png')
plt.show()
