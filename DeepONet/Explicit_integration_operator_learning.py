import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('../Python_libs')
import NNlibs
import random
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


N_m    = 50
A_dict = [1,2,4,6,8,10,12]
N_y    = 10
N_p    = 4
epoch  = 1000

criteria = nn.MSELoss()

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
#	plt.plot(x,U(x,i),'b',alpha=.4)
#	plt.plot(x,G_of_U(x,i),'r',alpha=.4)
#	plt.plot(x[idx],-2*np.ones(N_m),'rx')
	for k in range(0,N_y):
		random_idx = random.randint(0,len(x)-1)
		y = x[random_idx]
		g = G_of_U(x,i)[random_idx]
#		plt.plot(y,g,'ko')
		Sensor.append(U(x,i)[idx])
		Y.append(x[random_idx])
		G.append(g)
 	
#plt.xlabel('$x$', fontsize=18)
#plt.ylabel('$U$ and $G(U)$',fontsize=16)
plt.xlim(0,np.pi*2)
#plt.savefig('testimage.png',bbox_inches='tight')

### Convert numpy arrays to tensors

U_train = torch.tensor(np.array(Sensor),dtype=torch.float32)
Y_train = torch.tensor(np.array(Y).reshape(N_y*len(A_dict),1),dtype=torch.float32)
G_train = torch.tensor(np.array(G).reshape(N_y*len(A_dict),1),dtype=torch.float32)

#### Model Initialization and Training

DNN = NNlibs.DeepONet(branch_input=N_m,branch_num_layers=4,
	branch_layer_width=8,
	trunk_input=1,
	trunk_num_layers=4,trunk_layer_width=8,latent_layer=N_p)
optimizer = optim.Adam(DNN.parameters(), lr=0.001)

for i in range (0,epoch):
	y_pred = DNN(U_train,Y_train)
	y_trgt = G_train
	loss = criteria(y_pred,y_trgt)
	print('Epoch,  Loss: ', i,loss)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()


#### PREDICTION #######


U_pred = U(x,2)[idx]
plt.plot(x,U(x,2),'r')
plt.plot(x,G_of_U(x,2),'b')
plt.plot(x[idx],U(x,2)[idx],'kx')

x_pred = np.linspace(0,2*np.pi,500)

U_lst = []
for i in range (0,500):
	U_lst.append(U_pred)

U_pred = torch.tensor(np.array(U_lst),dtype=torch.float32)
Y_pred = torch.tensor(np.array(x_pred).reshape(500,1),dtype=torch.float32)

G_pred = DNN(U_pred,Y_pred)
G_pred = G_pred.detach().numpy().reshape(500)

plt.plot(x_pred, G_pred, 'k')
plt.xlabel('$x$',fontsize=18)
plt.ylabel('$G(U)$', fontsize=18)
plt.savefig('Prediction.png',bbox_inches='tight')
