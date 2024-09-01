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

import torch as torch
import torch.nn as nn
import Modules
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# Single layer Neural Networks perform linear regression of the form $t = b_0 + \sum_i w_i x$, where $x$ is the naive input variable or parameter. Magically if we know the non-linear inter-relationship  between these input parameters, we can create features or basis functions that can capture the correlation between the input space (prescribed) and between the input space and the output. There are several techniques for feature extraction using NN architecture. 
#
#
# Suppose we have a true generator $y = 3+4x+10x^2$. $N_p$ points are sampled from this true generator with a Gaussian noise of $\sigma$ as shown below:

# +
data          = Modules.Data_generator(3,4,10)

model0        = Modules.SingleLayer(1,1)
optimizer0    = torch.optim.Adam(model0.parameters(), lr =0.01)

model         = Modules.SingleLayer(5,1)
optimizer     = torch.optim.Adam(model.parameters(), lr =0.001)

model2        = Modules.SingleLayer(2,1)
optimizer2    = torch.optim.Adam(model2.parameters(), lr = 0.01)

loss_function = nn.MSELoss()


# +
def PolyBasis(x):
    temp  = np.zeros((np.size(x),5))
    for i in range (0,np.size(x)):
        temp[i,0] = x[i]**1
        temp[i,1] = x[i]**2
        temp[i,2] = x[i]**3
        temp[i,3] = x[i]**4
        temp[i,4] = x[i]**5
    return(temp)

def QuadraticBasis(x):
    temp  = np.zeros((np.size(x),2))
    for i in range (0,np.size(x)):
        temp[i,0] = x[i]**1
        temp[i,1] = x[i]**2
    return(temp)


# +
epoch  = 50000
Np     = 40
x_lo   = -1
x_hi   = 1
sigma  = 2

x_cont = torch.linspace(x_lo,x_hi,100)

# +
x_obs, y_obs = data.gaussian_sampling(x_lo,x_hi,Np,sigma)
y_true       = data.true_generator(x_cont)

x_features   = torch.tensor(PolyBasis(np.array(x_obs)),dtype=torch.float32)

# +
plt.plot(x_obs,y_obs, 'bo' , mfc = 'w', label='Observations')
plt.plot(x_cont, y_true, 'r',label='True generator')
plt.xlim(x_lo, x_hi)
plt.xlabel('$x$', fontsize=22)
plt.ylabel('$y$', fontsize=22)
plt.legend(fontsize=18, frameon=False)

plt.savefig('Images/Raw_data_for_Single_layer_NN.png')
# -

# ### Inability of a single layer Neural Network to approximate the true generator

# If we train a single layer neural network with the naive observed datapoints, the best we can expect is a single line that cuts through the data points as shown below:

for i in range (0,epoch):
    y_pred = model0(x_obs.reshape(Np,1))
    loss   = loss_function(y_pred.reshape(Np,1),y_obs.reshape(Np,1))
    loss.backward()
    optimizer0.step()
    optimizer0.zero_grad()

# +
plt.plot(x_obs,y_obs, 'bo' , mfc = 'w', label='Observations',alpha=0.4)
plt.plot(x_cont, y_true, 'r',label='True generator',alpha=0.4)


plt.plot(x_cont, model0(x_cont.reshape(100,1)).detach(),'b',label='Single layer NN')

plt.xlim(x_lo, x_hi)
plt.xlabel('$x$', fontsize=22)
plt.ylabel('$y$', fontsize=22)
plt.legend(fontsize=18, frameon=False)

plt.savefig('Images/Naive_data_in.png')
# -

# Next if we magically see that the are polynomial input features such as $\{ 1, x, x^2, x^3 , ... \}$. Lets cap it at $x^5$. Therefore, our feature space or basis is $\{ 1, x, x^2, x^3 , x^4 , x^5 \}$. Noe the input to the single layer Neural Network i sthe feature corresponding to the observed points $x_{obs}$. In some sense, we are building in more information into the network/regression.

for i in range(0,epoch):
    loss = loss_function(model.forward(x_features.reshape(Np,5)), y_obs.reshape(Np,1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# +

plt.plot(x_obs,y_obs, 'bo' , mfc = 'w', label='Observations',alpha=0.3)
plt.plot(x_cont, y_true, 'r',label='True generator',alpha=0.3)


plt.plot(x_cont, model0(x_cont.reshape(100,1)).detach(),'b',label='Single layer NN',alpha=.3)


x_basis = torch.tensor(PolyBasis(np.array(x_cont)),dtype=torch.float32)

plt.plot(x_cont,model(x_basis.reshape(100,5)).detach(), 'g',label='Single layer NN + basis')


plt.xlim(x_lo, x_hi)
plt.xlabel('$x$', fontsize=22)
plt.ylabel('$y$', fontsize=22)
plt.legend(fontsize=18, frameon=False)

plt.savefig('Images/Large_number_of_basis.png')
# -

# The presence of higher order terms in the basis lets the network overfit to outliers giving higher weigtage to the higher order terms. If we know the form of the underlying generator, which is usually not always, we can can eliminate this overfit. But this not always trivial. We will have to start with a complex model and 'regularize' the weights so that no mode/basis overfits the data

# +
x_feature =QuadraticBasis(np.array(x_obs))
x_feature = torch.tensor(x_feature, dtype = torch.float32)

for i in range (0,epoch):
    y_pred = model2(x_feature.reshape(Np,2))
    loss = loss_function(y_pred.reshape(Np,1), y_obs.reshape(Np,1))
    loss.backward()
    optimizer2.step()
    optimizer2.zero_grad()

# +
plt.plot(x_obs,y_obs, 'bo' , mfc = 'w', alpha=0.3)
plt.plot(x_cont, y_true, 'r',alpha=0.3)


plt.plot(x_cont, model0(x_cont.reshape(100,1)).detach(),'b',label='Single layer NN',alpha=.3)

x_basis = torch.tensor(PolyBasis(np.array(x_cont)),dtype=torch.float32)

plt.plot(x_cont,model(x_basis.reshape(100,5)).detach(), 'g',label='Single layer NN + basis')

x_basis = torch.tensor(QuadraticBasis(np.array(x_cont)),dtype=torch.float32)


plt.plot(x_cont,model2(x_basis.reshape(100,2)).detach(), 'orange',label='Single layer NN +  quadratic basis')


plt.xlim(x_lo, x_hi)
plt.xlabel('$x$', fontsize=22)
plt.ylabel('$y$', fontsize=22)
plt.legend(fontsize=18, frameon=False)

plt.savefig('Images/Quadratic_basis.png')
# -


