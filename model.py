import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_util

from torch import nn, LongTensor
from torch.autograd import Variable
from torch.nn import LSTM
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from tcn import TemporalConvNet


class Encoder_TCN(nn.Module):
    def __init__(self, input_size, output_size,
                 num_channels, kernel_size, dropout):
        """
        Sequential encoder with TCN as the main building block
        @param input_size: len of the input vector at each time step
        @param output_size: len of the output vector at each time step 
        @param num_channels: number of neurons as a list. e.g. [500, 1000, 1000, 500]
        @param kernel_size: length of the receptive field
        @param dropout: dropout; in tasks like this, it is usually effective to turn dropout to around 0.1
        
        The effective history (how many past timesteps can an output neuron look at)
        effective history length = (kernel_size - 1) * 2^len(num_channels)
        e.g.  kernel_size = 3, num_channels = [100,100,100,100] => history length = 32
        """
        super(Encoder_TCN, self).__init__()        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, inputs):
        """
        TemporalConvNet inputs have to have dimension (N, C_in, L_in)
        
        @param inputs: tensor with dimension (N, L_in, C_in)
        @return o: tensor with dimension (N, L_in, C_in)
        """
        o = self.tcn(inputs.transpose(1,2))  # input should have dimension (N, C, L)
        return o.transpose(2,1)
      
    
######################################################
#                  Wrapper Module
###################################################### 


class MLP(nn.Module):
    def __init__(self, in_dim, in_layers, out_dim, dropout=0.7, weight_tie=False, variational=True, device='cpu'):
        super(MLP, self).__init__()
        
        cuda_available = torch.cuda.is_available()
        self.device = device
            
        in_layer_dims = list(zip(in_layers, in_layers[1:]+[out_dim])) # set latent_dim to 2
        self.encoder = self.build_encoder(in_layer_dims, dropout)
        
        
    def build_encoder(self, in_layer_dims, dropout):
        encoder_layer = []

        for idx, (in_dim, out_dim) in enumerate(in_layer_dims):
            if idx > 0:
                encoder_layer.append(nn.ReLU())
                encoder_layer.append(nn.Dropout(dropout))
            encoder_layer.append(nn.Linear(in_dim, out_dim))
                    
        encoder = nn.Sequential(*encoder_layer)      
        return encoder
    
    
    def forward(self, inputs):
        # latent variable
        out = self.encoder(inputs)
        return out
    
    
    
class TCN_Deposit(nn.Module):
    
    def __init__(self, params):
        """
        @param features: list of features to use
        @param embed_dim: embedding dimension of the input tokens
        @param vocab_lens: vocab size of each categorical features
        @param mlp_layers: list of ints, hidden layers of mlp
        @param latent_dim: dim of the latent vector, the "embedded information"
        @param dropout: dropoutM
        @param layers: number of layers for LSTM
        @param hidden_units: number of hidden units in LSTM
        @param bias: initial forgetting gate bias.
        """
        super(TCN_Deposit, self).__init__()
        cuda_available = torch.cuda.is_available()
        self.device = params['device']
    
        if self.device != 'cpu':
            self.device = 'cuda:0'
        elif self.device == 'cpu' and cuda_available:
            print("cuda is available, you should use GPU")

        self.params = params

        # generate embeddings
        self.features = params['source_features'] 
        self.features_num = params['features_num']
        self.features_cat = params['features_cat']
        self.vocab_lens = params['vocab_lens']
        self.embed_dims = params['embed_dims']
        
        for f in list(self.vocab_lens.keys()):
            setattr(self, 'embeds_'+f, nn.Embedding(self.vocab_lens[f], self.embed_dims[f]))
                
        tot_embed_dim = sum(self.embed_dims.values()) + len(self.features_num)
        self.params['tot_embed_dim'] = tot_embed_dim
        
        seq_enc_param = {'input_size': params['tot_embed_dim'], 
                         'output_size': params['output_size'],
                         'num_channels': params['layers'], 
                         'kernel_size': params['kernel_size'],  # k = 4 -> history scope: (3-1) * 2^(4-1) = 32
                         'dropout': params['dropout_tcn']}

        mlp_param = {'in_dim': seq_enc_param['output_size'],
                     'in_layers': params['in_layers'],
                     'out_dim': params['output_size'],
                     'dropout': params['dropout_mlp']}
        
        
        self.seq_encoder = Encoder_TCN(**seq_enc_param)
        self.linear = MLP(**mlp_param)

        print("\n\n INITIALIZING A NEW MODEL \n\n\n\n")
        
        
    def embed_inputs(self, inputs):
        # self.features contains all input features
        # vocab_lens and embed_im only contains features that requires embedding
        # inputs needs to be integer encoded
        # inputs has dim: batch x seq_len x features -> batch x seq_len x features_embed_tot
        
        b, s, d = inputs.shape
        inputs_l = []
        
        for i in range(len(self.features)):
            feature = self.features[i]
            if feature in self.features_cat:
                embedding = getattr(self, 'embeds_'+feature)
                inputs_l.append(embedding(inputs[:, :, i].long()))
                
            elif feature in self.features_num: 
                inputs_l.append(inputs[:, :, i].reshape(b, s, 1).float())
        
        inputs = torch.cat(tuple(inputs_l), dim=2)
        return inputs
    

    def forward(self, inputs): 
        out = self.embed_inputs(inputs)
        out = self.seq_encoder(out)
        out = self.linear(out)
        return out