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

# +
x_cont = torch.linspace(-1,1,100)
x_lo   = -1
x_hi   =  1
Np     = 40 
sigma  = 1

n_in   = 1
n_hid  = 100
n_hid1 = 10
n_hid2 = 10
n_out  = 1

epoch  = 50000
# -

model_1       = Modules.SLNN(n_in, n_hid, n_out)
model_2       = Modules.TLNN(n_in, n_hid1, n_hid2, n_out)
loss_function = nn.MSELoss()
optimizer_1   = torch.optim.SGD(model_1.parameters(), lr=0.01)
optimizer_2   = torch.optim.SGD(model_2.parameters(), lr=0.01)

# Suppose data a true data generator is given as $y = 2+3x+6x^2$, while the observed data is Gaussian about the true  generator with a Gaussian noise with standard deviation $\sigma$. We use a single layer neural network of hidden dimension 100 and a Two layer neural network with 10 $\times$ 10 hidden units across two layers to model the true generator (conditional probability). 

data = Modules.Data_generator(2,3,6)

# +
y_true = data.true_generator(x_cont)
x_obs,y_obs = data.gaussian_sampling(x_lo,x_hi,Np,sigma)
plt.plot(x_obs, y_obs,'ro', mfc='w', markersize=8)
plt.plot(x_cont, y_true,'b')
plt.xlim(x_lo,x_hi)

plt.xlabel('$x$',fontsize=22)
plt.ylabel('$y$',fontsize=22)
plt.savefig('Images/true_generator_and_observed_data.png')
# -

# Reshape the input and output tensors to meet the model dimensions

x_obs = x_obs.reshape(Np,n_in)
y_obs = y_obs.reshape(n_out,Np)

_loss_1 = []
_loss_2 = []
for i in range (0,epoch):
    y_pred_1 = model_1.forward(x_obs)
    loss1   = loss_function(y_pred_1,y_obs.reshape(y_pred_1.size()))
    _loss_1.append(loss1.detach())
    loss1.backward()
    optimizer_1.step()
    optimizer_1.zero_grad()

    y_pred_2 = model_2.forward(x_obs)
    loss2   = loss_function(y_pred_2,y_obs.reshape(y_pred_2.size()))
    _loss_2.append(loss2.detach())
    loss2.backward()
    optimizer_2.step()
    optimizer_2.zero_grad()

# +
plt.plot(_loss_1,'m',label ='Single layer')
plt.plot(_loss_2,'g',label = 'Two layers')
plt.xlim(0,epoch/10)
plt.xlabel('epoch',fontsize=18)
plt.ylabel('$||\\tilde{y} - y||_2^2$',fontsize=18)
plt.legend(fontsize=22,frameon=False)

plt.savefig('Images/Single_vs_double_lossVsEpoch.png')

# +
plt.plot(x_obs.reshape(Np), y_obs.reshape(Np),'ro', mfc='w', markersize=8,alpha=0.3)
plt.plot(x_cont, y_true,'b',alpha=.4)
plt.plot(x_cont.reshape(100,1),model_1.forward(x_cont.reshape(100,1)).detach(),'magenta',label = 'Single layer')
plt.plot(x_cont.reshape(100,1),model_2.forward(x_cont.reshape(100,1)).detach(),'green',label = 'Two layers')
plt.xlim(x_lo,x_hi)

plt.xlabel('$x$',fontsize=22)
plt.ylabel('$y$',fontsize=22)
plt.legend(fontsize=22,frameon=False)

plt.savefig('Images/Single_layer_two_layer_comparison.png')
# -

model_1.Layer1.weight.size()


