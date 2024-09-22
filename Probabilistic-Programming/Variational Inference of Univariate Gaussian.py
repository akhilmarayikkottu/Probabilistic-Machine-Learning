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

import pyro 
import numpy as np
import matplotlib.pyplot as plt
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import Modules
import torch

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# +
mu  = 0.0
std = 1
Np  = 2000
epoch = 50000

x_lo  = -3
x_hi  =  3 

x_cont = np.linspace(x_lo,x_hi,1000)
# -

img_dir = 'Images/'

X_obs = np.random.normal(mu, std, Np)

# +
plt.figure(figsize=(8,5))

plt.plot(x_cont,Modules.UniGaussian(mu,std,x_cont),'b')
plt.plot(X_obs, np.ones(Np)*0,'ro', mfc ='w', label='$x_{obs}$',alpha=0.1,markersize=12)

plt.xlabel('$x$', fontsize=22)
plt.ylabel('$\mathcal{N}(x| \mu, \sigma)$',fontsize=20)
plt.legend(fontsize=22,frameon=False)
plt.xlim(x_lo,x_hi)

plt.savefig(img_dir+'Observed_data.png')


# -

def model(x):
    std = pyro.sample('std dev', dist.Gamma(0.5,0.5))
    mu  = pyro.sample('mu' , dist.Normal(0,3))
    with pyro.plate('obs', len(x)):
        out = pyro.sample("x", dist.Normal(mu,std), obs=x)
    return(out)


x = torch.tensor(X_obs, dtype=torch.float)
pyro.render_model(model,model_args=(x,),render_distributions=True,render_params=True,filename=img_dir+'univariatemodel.png')


def guide(x):
    alpha = pyro.param('alpha', lambda: torch.rand(()) , constraint=constraints.positive)
    beta  = pyro.param('beta' , lambda: torch.rand(()) , constraint=constraints.positive)
    std   = pyro.sample('std dev', dist.Gamma(alpha,beta))
    
    mu0  = pyro.param('mu 0', lambda: torch.rand(()) )
    sigma0 =  pyro.param('sigma 0', lambda: torch.rand(()) , constraint=constraints.positive)
    mu =  pyro.sample('mu', dist.Normal(mu0,sigma0))
    return{"mu" : mu, 'std dev': std}
    


pyro.render_model(guide,model_args=(x,),render_distributions=True,render_params=True,filename=img_dir+'unigausguide.png')

adam = pyro.optim.Adam({"lr": 0.02})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)

pyro.clear_param_store()
losses = []
for step in range(0,epoch):  
    loss = svi.step(x)
    losses.append(loss)

# +
plt.figure(figsize=(8,4))
plt.semilogy(losses,'k',linewidth=.5)
plt.xlabel("epoch", fontsize=18)
plt.ylabel("$\mathcal{L} \Bigg(q(Z) \Bigg)$", fontsize=16)
plt.xlim(0,epoch)

plt.savefig(img_dir+'unigauss_conv.png')
# -

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())

predictive = pyro.infer.Predictive(model, guide=guide, num_samples=200000)
svi_samples = predictive(x)

mu2  = svi_samples["mu"].detach().numpy().reshape(-1,)
std2 = svi_samples["std dev"].detach().numpy().reshape(-1,)

# +
plt.figure(figsize=(6,6))

plt.hist2d(mu2,std2,bins=(50, 50),cmap='jet')
#plt.xlim(-.75,.75)
#plt.ylim(.0,40.)
plt.xlabel('$\mu$',fontsize=22)
plt.ylabel('$\sigma$',fontsize=22)
plt.title('$p(\mu, \sigma | X)$', fontsize=22)
plt.savefig(img_dir+'posterior_unigauss.png')
# -


