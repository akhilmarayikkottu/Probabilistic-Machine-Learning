import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('../Python_libs')
import NNlibs
import random

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


N_m = 10
A_dict = [1,2,4]
N_y = 10

#data = np.zeros((N_y*len(A_dict),N_m+2))

Sensor = []
Y      = []
G      = []
def U(x,A,C=0):
	tmp = np.sin(A*x)+C*x 
	return (tmp)
def G_of_U(x,A,C=0):
	tmp = -1*np.cos(A*x)/A+C*x*x/2
	return tmp

x = np.linspace(0,2*np.pi,1000)

idx = np.round(np.linspace(0, len(x) - 1, N_m)).astype(int)

plt.figure(figsize=(8,5))
for i in  A_dict:
	plt.plot(x,U(x,i),'b',alpha=.4)
	plt.plot(x,G_of_U(x,i),'r',alpha=.4)
	plt.plot(x[idx],-2*np.ones(N_m),'rx')
	for k in range(0,N_y):
		random_idx = random.randint(0,len(x)-1)
		y = x[random_idx]
		g = G_of_U(x,i)[random_idx]
		plt.plot(y,g,'ko')
		Sensor.append(U(x,i)[idx])
		Y.append(x[random_idx])
		G.append(g)
 	
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$U$ and $G(U)$',fontsize=16)
plt.xlim(0,np.pi*2)
plt.savefig('testimage.png',bbox_inches='tight')

### Convert numpy arrays to tensors

U = torch.tensor(np.array(Sensor),dtype=torch.float32)
Y = torch.tensor(np.array(Y).reshape(N_y*len(A_dict),1),dtype=torch.float32)
G = torch.tensor(np.array(G).reshape(N_y*len(A_dict),1),dtype=torch.float32)

print(G)

#### Model Initialization and Training

DNN = NNlibs.DeepONet(branch_input=10,trunk_input=1,latent_layer=8)

DNN(U,Y)
