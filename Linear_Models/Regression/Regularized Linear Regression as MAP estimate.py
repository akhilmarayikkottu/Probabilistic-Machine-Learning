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

import functions
import numpy as np
import random
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


def RegModelMatrix(A,lambd):
    Phi = np.matmul(A.T,A)
    return (Phi+lambd*np.identity(6))


# +
x = np.linspace(-2,2,1000)

obs_x,obs_t  = functions.noisy_obs(-2,2,100,10)

plt.plot(x,functions.true_output(x),'b')
plt.plot(obs_x,obs_t, 'rx')
plt.xlim(-2,2)
plt.ylabel('$t$',fontsize=22); plt.xlabel('$x$',fontsize=22)
plt.savefig('Images/Raw_data2.png')
plt.show()

# +
A = functions.ModelMatrix(np.size(obs_x),obs_x)
Phi = RegModelMatrix(A,1.0)
Phi_rhs = np.matmul(A.T,obs_t)

Phi_inv = np.linalg.inv(Phi)

a,b,c,d,e,f = np.matmul(Phi_inv, Phi_rhs)


# +
lambdas = [.001, .1, 1, 10, 100]
plt.plot(x,functions.true_output(x),'b')
plt.plot(obs_x,obs_t, 'rx')
for i in lambdas:
    Phi = RegModelMatrix(A,i)
    Phi_inv = np.linalg.inv(Phi)
    a,b,c,d,e,f = np.matmul(Phi_inv, Phi_rhs)
    plt.plot(x,functions.poly_model(a,b,c,d,e,f,x),label=str(i))
    
plt.xlim(-2,2)
plt.ylabel('$t$',fontsize=22); plt.xlabel('$x$',fontsize=22)
plt.legend(fontsize=16,frameon=False)
plt.savefig('Images/MAP_out.png')
plt.show()
