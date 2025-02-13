import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
plt.rcParams['text.usetex'] = True
import sys
sys.path.append('/nfs/home/6/marayika/Common_scripts')
import Module_NN
from torch.utils.data import DataLoader, TensorDataset


N_m    = 1000
A_dict = [1,2,3,4,5,6,7,8]
N_y    = 500
N_p    = 4
epoch  = 1000
branch_layers = 2
trunk_layers  = 2
batch_size = 6
branch_width = 25
trunk_width  = 25

criteria = nn.MSELoss()

Sensor = []
Y      = []
G      = []

def U1(x,A):
    tmp = x*x+A
    return tmp

def G_of_U1(x,A):
    tmp = x*x*x+A*x
    return tmp

def U(x,A,C=0):
	tmp = np.sin(A*x)+C*x 
	return (tmp)
def G_of_U(x,A,C=0):
	tmp = -1*np.cos(A*x)/A+C*x*x/2
	return tmp

x = np.linspace(0,2*np.pi,1000)

idx = np.round(np.linspace(0, len(x) - 1, N_m)).astype(int)

fig,ax = plt.subplots(1,2,figsize=(16,5))
for i in  A_dict:
    ax[0].plot(x,U(x,i),'b',alpha=.4)
    ax[0].plot(x,G_of_U(x,i),'r',alpha=.4)
    for k in range(0,N_y):
        random_idx = random.randint(0,len(x)-1)
        y = x[random_idx]
        g = G_of_U(x,i)[random_idx]
#		plt.plot(y,g,'ko')
        Sensor.append(U(x,i)[idx])
        Y.append(x[random_idx])
        G.append(g)
 	
ax[0].set_xlabel('$x$', fontsize=18)
ax[0].set_ylabel('$U$ and $G(U)$',fontsize=16)
ax[0].set_xlim(0,np.pi*2)
#plt.savefig('testimage.png',bbox_inches='tight')

### Convert numpy arrays to tensors

U_train = torch.tensor(np.array(Sensor),dtype=torch.float32)
Y_train = torch.tensor(np.array(Y).reshape(N_y*len(A_dict),1),dtype=torch.float32)
G_train = torch.tensor(np.array(G).reshape(N_y*len(A_dict),1),dtype=torch.float32)

dataset = TensorDataset(U_train,Y_train,G_train)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle =True)

#### Model Initialization and Training

DNN = Module_NN.DeepONet(branch_input=N_m,branch_num_layers=branch_layers,
	branch_layer_width=branch_width,
	trunk_input=1,
	trunk_num_layers=trunk_layers,
    trunk_layer_width=trunk_width,
    latent_layer=N_p,
    activation=nn.Sigmoid(),
    branch_bias = True, trunk_bias = True)
optimizer = optim.Adam(DNN.parameters(), lr=0.0001)

print(DNN)

for i in tqdm(range (0,epoch)):
    for l,m,n in dataloader:
        y_pred = DNN(l,m)
        y_trgt = n
        loss = criteria(y_pred,y_trgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


#### PREDICTION #######


U_pred = U(x,4)[idx]
plt.plot(x,U(x,4),'r')
plt.plot(x,G_of_U(x,4),'b',label='target')

x_pred = np.linspace(0,2*np.pi,500)

U_lst = []
for i in range (0,500):
	U_lst.append(U_pred)


U_pred = torch.tensor(np.array(U_lst),dtype=torch.float32)
Y_pred = torch.tensor(np.array(x_pred).reshape(500,1),dtype=torch.float32)

G_pred = DNN(U_pred,Y_pred)
G_pred = G_pred.detach().numpy().reshape(500)



ax[1].plot(x_pred, G_pred, 'k',label='prediction')
ax[1].set_xlabel('$x$',fontsize=18)
ax[1].legend(frameon=False,fontsize=16)
plt.savefig('Prediction.png',bbox_inches='tight')
