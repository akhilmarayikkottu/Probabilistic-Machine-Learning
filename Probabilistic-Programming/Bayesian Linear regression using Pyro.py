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
import pyro.distributions as dist
import matplotlib.pyplot as plt
import torch
import Modules
import numpy as np
import pyro.distributions.constraints as constraints

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# Let us assume that we have a noisy data sampled from a tru generator $t = 10x+2$ as shown in the figure below and previously used in the un-regularized linear regression. In this section, we try to give this MLE prediction a probabilistic overtone. We can assume that the output generated from the noisy generator follows a Gaussian conditional distribution $\mathcal{N} (t| \mu , \sigma)$, here the mean of the distribution is some function of the form $\mu = Ax+B$ where $A$ and $B$ are learning parameters and together the model form a class of functions. $\sigma$ is a learnable parameter which describes the spread of the datapoints with respect to the mean $\mu$.

epoch = 5000

# +
x_cont = np.linspace(-2,2,1000)

obs_x,obs_t  = Modules.noisy_obs(-2,2,100,10)

plt.plot(x_cont,Modules.true_output(x_cont),'b', label="True generator")
plt.plot(obs_x,obs_t, 'rx',label="Observed data")
plt.xlim(-2,2)
plt.ylabel('$t$',fontsize=22); plt.xlabel('$x$',fontsize=22)
plt.legend(fontsize=16, frameon=False)
plt.savefig('Images/Raw_data.png')
plt.show()


# -

# We have to define a probabilistic model of the form $p(t|x) = \mathcal{N}(t| Ax+B, \sigma)$. Using regular numpy libraries, as shown before, the fucntion can be casta s a least-square optimization problem wby taking logrithm of the distribution and maximizing the hence obtained convex function. Here (fortunately !) we cast this problem as a fully-probabilistic model using the Pyro-ppl package (a probabilistic layer over pytorch). The formulation shown below as a graphical model generates or samples data points from the above mentioned distribution. Here, the parameters are deterministic and the graph represents an MLE approximation for the model.

def model(x,t):
    A     = pyro.param("A", lambda: torch.rand(()) )
    B     = pyro.param("B", lambda: torch.rand(()))
    sigma = pyro.param("sigma", lambda: torch.ones(()), constraint=constraints.positive)
    mean  = A*x+B
    with pyro.plate("Data", len(x)):      
        C = pyro.sample("t", dist.Normal(mean, sigma), obs =t)
    return(C)


x = torch.tensor(obs_x, dtype=torch.float)
t = torch.tensor(obs_t, dtype=torch.float)
pyro.render_model(model,model_args=(x, t),render_distributions=True,render_params=True,filename="Images/MLEgraph.png")


# To convert the model to incorporate the uncertainty in the latent parameters $Z = \{ A,B, \sigma \}$ we convert these parameters to (parametric) distributions which can capture their uncertainities. Given the output is $t = \mathcal{N}(t|Ax+B, \sigma)$,  $\sigma \in (0, \infty)$ while $A$ and $B$ $\in (- \infty, \infty)$. To build in this constraints, $A$ and $B$ distributions are assumed to be Gaussians while $\sigma$ is assumed to be a uniform distribution $(0, \infty)$. The graphical model for this bayesian model is : 

def model(x,t):
    A     = pyro.sample("A", dist.Normal(0,10) )
    B     = pyro.sample("B", dist.Normal(0,10))
    sigma = pyro.sample("sigma", dist.Uniform(0,50))
    mean  = A*x+B
    with pyro.plate("Data", len(x)):      
        C = pyro.sample("t", dist.Normal(mean, sigma), obs =t)
    return(C)


x = torch.tensor(obs_x, dtype=torch.float)
t = torch.tensor(obs_t, dtype=torch.float)
pyro.render_model(model,model_args=(x, t),render_distributions=True,render_params=True,filename="Images/Bayesiangraph.png")

# For simplicity, we infer a guide (variational distribution) $q(Z)$ as non-correlated distributions generating the latent space $Z$ or the weight space. As shown below, $A$, $B$ and $\sigma$ are uncorrelated. This {\it guide} is not too flexible to capture the correlations between the parameters iteself (Not a bad approximation though !!!).

auto_guide = pyro.infer.autoguide.AutoNormal(model)

pyro.render_model(auto_guide,model_args=(x, t),render_distributions=True,filename="Images/guide.png")

# In the inference stage, the variational distribution $q(Z)$ parameters (parameters for the factorized distributions building the ${\it guide}$ ) through varitional inference (Here stochastic varaitional inference). SVI is a power house and the usp of Pyro-ppl !!

adam = pyro.optim.Adam({"lr": 0.02})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

pyro.clear_param_store()
losses = []
for step in range(epoch):  
    loss = svi.step(x, t)
    losses.append(loss)
    #if step % 100 == 0:
    #    logging.info("Elbo loss: {}".format(loss))

# +
plt.figure(figsize=(8,4))
plt.plot(losses,'k',linewidth=.5)
plt.xlabel("epoch", fontsize=18)
plt.ylabel("$\mathcal{L} \Bigg(q(Z) \Bigg)$", fontsize=16)

plt.xlim(0,epoch)
plt.savefig("Images/ELBO.png")
# -

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())

predictive = pyro.infer.Predictive(model, guide=auto_guide, num_samples=10000)
svi_samples = predictive(x, t=None)

# The trained ${\it guide}$, can be used to see the variation in the weight space. It could also indicate the correlations between these parameters. Since I was super lazy, I went with a diagonal covariance matrix in the ${\it guide}$ imposing all three parameters to be independent of each other. Joint probability will just give a product contour of these individual distributions!

# +
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4), sharey=True)

A_bins = np.linspace(5,15,100)
ax[0].hist(svi_samples["A"].detach().numpy(),color='b',alpha=.3,density = True, bins=A_bins)
ax[0].set_xlabel("$A$", fontsize=18)
ax[0].set_ylabel("$p(A)$", fontsize=18)

B_bins = np.linspace(-1,7,100)
ax[1].hist(svi_samples["B"].detach().numpy(),color='b',alpha=.3,density=True,bins=B_bins)
ax[1].set_xlabel("$B$", fontsize=18)
ax[1].set_ylabel("$p(B)$", fontsize=18)

sigma_bins = np.linspace(6,14,100)
ax[2].hist(svi_samples["sigma"].detach().numpy(),color='b',alpha=.3,density=True, bins= sigma_bins)
ax[2].set_xlabel("$\sigma$", fontsize=18)
ax[2].set_ylabel("$p(\sigma)$", fontsize=18)

plt.savefig("Images/UncertaintyInParameters.png")
# -

predictive = pyro.infer.Predictive(model, guide=auto_guide, num_samples=1)
svi_samples = predictive(x, t=None) 

# VOILA!!! We have a generative model. Ancestral sampling is the way to go. sampling generates distributions of the form $\tilde{t} = \mathcal{N}(\tilde{t}| \tilde{\mu}, \tilde{\sigma})$. where, $\tilde{\mu}$ and $\tilde{\sigma}$ are inturn random variables generated from parents.

plt.plot(x.detach().numpy(),svi_samples["t"].detach().numpy().reshape(100),'rx',alpha=.4,label="Observed data")
plt.plot(x_cont,Modules.true_output(x_cont),'b',alpha =.5, label="True generator")
plt.plot(obs_x,obs_t, 'ko',mfc='w', label="Generated data")
plt.legend(fontsize=16, frameon=False)
plt.xlim(-2,2)
plt.ylabel('$t$',fontsize=22); plt.xlabel('$x$',fontsize=22)
plt.savefig("Images/GeneratedData.png")


