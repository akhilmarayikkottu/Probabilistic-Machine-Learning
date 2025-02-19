import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

class FNN(nn.Module):
    
    def __init__(self,list_layers, act, bias=True,device=None):
        super(FNN,self).__init__()
        self.depth      = len(list_layers)
        self.activation = act
        self.bias       = bias
        self.device     = device
        
        self.layers = []
        for i in range (0,self.depth-1):
            self.layers.append(nn.Linear(list_layers[i],list_layers[i+1],bias,device=device))
            if (i != self.depth-2):
                if (self.activation == "ReLU"):
                    self.layers.append(nn.ReLU())
                if (self.activation == "Sigmoid"):
                    self.layers.append(nn.Sigmoid())
                if (self.activation == "Tanh"):
                    self.layers.append(nn.Tanh())             
                
    def forward(self,x):
        out = []
        out.append(x)
        for i in range(0,2*self.depth-3):
            tmp = self.layers[i](out[-1])
            out.append(tmp)
        ret = out[-1]
        return(ret)

class PandasDataset(Dataset):
    def __init__(self, dataframe, features, target):
        self.dataframe = dataframe
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        x = torch.tensor(row[self.features].values, dtype=torch.float32)
        y = torch.tensor(row[self.target].values, dtype=torch.float32)
        return x, y

import torch
from typing import List, Optional, Union, Any, Callable, Dict

class MLP(torch.nn.Module):
    def __init__(self,in_ftrs:int,out_ftrs:int,
                 n_layers:int=1,
                 layer_width:int = 25,
                 bias:bool=True,
                 final_layer_bias:bool = True,
                 activation: Optional[Any]=None,
                 final_layer_activation: Optional[Any]=None,):

        super().__init__()
        self.in_ftrs = in_ftrs
        self.out_ftrs = out_ftrs
        self.n_layers = n_layers
        self.layer_width = layer_width
        self.bias = bias
        self.final_layer_bias = final_layer_bias
        self.activation = activation
        self.final_layer_activation = final_layer_activation
        # Adding the first layer
        self.module_list = torch.nn.ModuleList([
                            torch.nn.Linear(self.in_ftrs,
                             self.layer_width,bias=self.bias)])
        # Adding non-linearity of first layer 
        if self.activation is not None:
            self.module_list.append(self.activation)
        # Intermediate layers & non-linearity
        for i in range(1,self.n_layers):
            self.module_list.append(
                  torch.nn.Linear(self.layer_width,
                                  self.layer_width,
                                  bias=self.bias))
            if self.activation is not None:
                self.module_list.append(
                                 self.activation)
        # Adding final layer
        self.module_list.append(torch.nn.Linear(self.layer_width,
                                    self.out_ftrs,
                                    bias=self.final_layer_bias))
        # Adding non-linearity of final layer
        if self.final_layer_activation is not None:
            self.module_list.append(
                       self.final_layer_activation)

    def forward(self,x):
        # Apply each layer
        for lyr in self.module_list:
            x = lyr(x)

        return x


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (abs(validation_loss-train_loss) ) < self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True



class AutoEncoder(torch.nn.Module):
    def __init__(self,in_ftrs:int,
            latent_width:int,
            n_layers_encoder:int=1,
            layer_width_encoder:int=25,
            n_layers_decoder:int=1,
            layer_width_decoder:int=25,
            bias:bool=True,
            activation: Optional[Any]=None):

        super().__init__()
        self.in_ftrs = in_ftrs
        self.n_layers_encoder = n_layers_encoder
        self.layer_width_encoder = layer_width_encoder
        self.latent_width    = latent_width
        self.n_layers_decoder = n_layers_decoder
        self.layer_width_decoder = layer_width_decoder
        self.bias = bias
        self.activation =activation

        # ENCODER
        module_lst_encoder = []
        ## Adding the first layer
        module_lst_encoder.append(torch.nn.Linear(self.in_ftrs,
            self.layer_width_encoder,
            bias=self.bias))
        ## Adding non-linearity of first layer
        if (self.activation is not None):
            module_lst_encoder.append(self.activation)
        ## Intermediate layersand activations
        for i in range (1,self.n_layers_encoder):
            module_lst_encoder.append(torch.nn.Linear(self.layer_width_encoder,
                self.layer_width_encoder,bias=sel.bias))
            if (self.activation is not None):
                module_lst_encoder.append(self.activation)
        ## Adding final layer that maps to latent space
        module_lst_encoder.append(torch.nn.Linear(self.layer_width_encoder,
            self.latent_width,bias=self.bias))

        self.encoder = nn.Sequential(*module_lst_encoder)

        # DECODER
        module_lst_decoder = []
        ## Adding first layer from latent space
        module_lst_decoder.append(torch.nn.Linear(self.latent_width,
            self.layer_width_decoder,bias=self.bias))
        ## Adding activation to the first layer 
        if (self.activation is not None):
            module_lst_decoder.append(self.activation)
        ## Adding intermediate layers and activations
        for i in range (1,self.n_layers_decoder):
            module_lst_decoder.append(torch.nn.Linear(self.layer_width_decoder,
                self.layer_width_decoder,bias=self.bias))
            if (self.activation is not None):
                module_lst_decoder.append(self.activation)
        ## Adding the final layer that maps to input
        module_lst_decoder.append(torch.nn.Linear(self.layer_width_decoder,
            self.in_ftrs,bias=self.bias))


        self.decoder = nn.Sequential(*module_lst_decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




class Trainer:
    def __init__(self, model, optimizer, criterion, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, train_loader: DataLoader, epochs: int, model_save_int:int,
            print_loss_int:int,val_loader: DataLoader = None, early_stop =None):

        _training_loss   = []
        _validation_loss = []

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

            _training_loss.append(train_loss)
            
            if val_loader:
                _validation_loss.append(self.validate(val_loader))

            if (epoch%print_loss_int == 0):
                self.loss_plot(_training_loss,_validation_loss,epochs,epoch)

            if (epoch%model_save_int == 0):
                self.save_model(epoch=epoch)

            if early_stop:
                if val_loader is not None:
                    early_stop(_training_loss[-1],_validation_loss[-1])
                    if early_stop.early_stop:
                        self.loss_plot(_training_loss, _validation_loss , epochs,epoch,estop=True)
                        self.save_model(epoch=epoch,estop=True)
                        print("Early stopping training loop")
                        break



    def validate(self, val_loader: DataLoader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        return (val_loss)

    def loss_plot(self,train ,valid,epochs, epoch, estop:bool=False):
        plt.figure(figsize=(8,4))
        plt.plot(train,'r')
        plt.plot(valid, 'b')
        plt.xlim(0,epochs)
        if estop == False:
            plt.savefig('loss_at_'+str(epoch)+'.png', bbox_inches='tight')

        if estop == True:
            plt.savefig('loss_at_'+str(epoch)+'_with_early_stop.png', bbox_inches='tight')

    def save_model(self,epoch,estop:bool=False):
        model_name = 'model_at_'+str(epoch)+'.pt'
        if estop:
            model_name = 'model_at_'+str(epoch)+'_with_early_stop.pt'
        model_tmp = torch.jit.script(self.model)
        model_tmp.save(model_name)


class DeepONet(torch.nn.Module):
    def __init__(self, branch_input:int,
            trunk_input:int,
            latent_layer:int,
            branch_num_layers:int = 1,
            branch_layer_width:int = 25,
            trunk_num_layers:int = 1,
            trunk_layer_width:int = 25,
            branch_bias:bool =False,
            trunk_bias:bool = False,
            combi_bias:bool = False,
            activation_at_trunk_end = nn.ReLU(),
            activation: Optional[Any]=None):


        super().__init__()
        self.m = branch_input
        self.y = trunk_input
        self.p = latent_layer
        self.N_b = branch_num_layers
        self.W_b = branch_layer_width
        self.N_t = trunk_num_layers
        self.W_t = trunk_layer_width
        self.bias_b = branch_bias
        self.bias_t = trunk_bias
        self.activation = activation
        self.act_trunk_end = activation_at_trunk_end
        # BRANCH NETWORK
        module_lst_branch = []
        ## Adding the first layer
        module_lst_branch.append(torch.nn.Linear(self.m,self.W_b,
            bias=self.bias_b))
        ## Adding non-linearity of first layer
        if (self.activation is not None):
            module_lst_branch.append(self.activation)
        ## Adding intermediate layers
        for i in range (1,self.N_b):
            module_lst_branch.append(torch.nn.Linear(self.W_b,self.W_b,
                bias=self.bias_b))
            if(self.activation is not None):
                module_lst_branch.append(self.activation)
        ## Adding final layer 
        module_lst_branch.append(torch.nn.Linear(self.W_b,self.p,
            bias=self.bias_b))

        self.branch = nn.Sequential(*module_lst_branch)

        # TRUNK NETWORK
        module_lst_trunk = []
        ## Adding the first layer
        module_lst_trunk.append(torch.nn.Linear(self.y,self.W_t,
            bias=self.bias_t))
        ## Adding activation to the first layer
        if (self.activation is not None):
            module_lst_trunk.append(self.activation)
        ## Adding intermediate layers
        for i in range (1,self.N_t):
            module_lst_trunk.append(torch.nn.Linear(self.W_t,self.W_t
                ,bias=self.bias_t))
            if (self.activation is not None):
                module_lst_trunk.append(self.activation)
        ## Adding final layer 
        module_lst_trunk.append(torch.nn.Linear(self.W_t,self.p,
            bias=self.bias_t))
        ## Adding activation function at the end
        module_lst_trunk.append(self.act_trunk_end)

        self.trunk = nn.Sequential(*module_lst_trunk)

    def forward(self, u, x):
        branch_output = self.branch(u)
        trunk_output  = self.trunk(x)

        output = torch.sum(branch_output*trunk_output,dim=-1, keepdim=True)
        return(output)

class Loss(nn.Module):
	def cvae_loss(recon_x, x, mu, logvar):
		# Reconstruction loss (BCE or MSE)
		recon_loss = F.mse_loss(recon_x, x, reduction='sum')  # Use BCE if inputs are normalized
		# KL divergence loss
		kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return recon_loss + kl_loss



class CVAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=16, 
	hidden_channels=[32, 64, 128], input_size=32,
	kernel_size:int = 3, stride:int = 2, padding:int = 1,
	BatchNormalization:bool = False, 
	activation:Optional[Any]=nn.ELU()):

        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels.copy()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.BatchNorm = BatchNormalization

        # Encoder
        layers = []
        in_channels = input_channels
        for out_channels in hidden_channels:
            layers.append(nn.Conv2d(in_channels, out_channels,kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            if(self.BatchNorm): layers.append(nn.BatchNorm2d(out_channels)) 
            layers.append(self.activation)
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        
        # Compute output size for FC layers dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, input_size, input_size)
            dummy_output = self.encoder(dummy_input)
            self.fc_input_dim = dummy_output.view(1, -1).size(1)
            self.feature_map_shape = dummy_output.shape[1:]  # Store (C, H, W) for reshaping
        
        # Latent space
        self.fc_mu = nn.Linear(self.fc_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.fc_input_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.fc_input_dim)
        
        # Decoder
        hidden_channels.reverse()
        layers = []
        in_channels = hidden_channels[0]
        for out_channels in hidden_channels[1:]:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.padding))
            if(self.BatchNorm): layers.append(nn.BatchNorm2d(out_channels))  # Adding BatchNorm
            layers.append(self.activation)
            in_channels = out_channels
        
        layers.append(nn.ConvTranspose2d(in_channels, input_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.padding))
        layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), *self.feature_map_shape)  # Use stored feature map shape
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar



