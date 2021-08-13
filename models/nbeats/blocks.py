from typing import Tuple
import math

from typing import Tuple, List
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NBeatsBlockBase(torch.nn.Module):

    def __init__(self,
              input_size,
              input_dim,
              theta_size: int,          
              basis_function: torch.nn.Module,
              output_dim = None,
              linear_layers: int = 4,
              layer_size: int = 512,
              attention_layers : int = 1,
              attention_embedding_size = 100,
              attention_heads = 1,
              positional_encoding=False,
              dropout = 0.1
              ):
        """
        N-BEATS block.
        :param input_size: length of input time series S.
        :param input_dim: input time series dimensionality D.
        :param output_dim: output time series dimensionality, differs from input_d only if additional time series features like lag features are created
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param linear_layers: Number of  linear layers.
        :param layer_size: Layer size of linear layer.
        :param attention_layers:  not needed here but still here that all block types have the same signature
        :param attention_embedding_size: embedding size of attention layers
        :param attention_heads: number of attention heads in attention layers
        :param positional_encoding: if true, using positional encoding in time attention block
        :param dropout: dropout used in all layers like attention, transformer encoder, positional encoding
        """
        super().__init__()
        self.input_size = input_size
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.theta_size = theta_size
        self.basis_function = basis_function
        self.linear_layers = linear_layers
        self.layer_size = layer_size
        self.attention_layers = attention_layers 
        self.attention_embedding_size = attention_embedding_size
        self.attention_heads = attention_heads
        self.use_positional_encoding = positional_encoding
        self.dropout = dropout

    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def init_block(self):
        raise NotImplementedError()



class NBeatsBlock(NBeatsBlockBase):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
                
        
        self.init_block()
    
    def init_block(self):
        # general Linear: (N,∗,H_in) --> (N,∗,H_out)
        # linear timeInputLayer: (N, E, S) --> (N, E, 512)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_features=self.input_size, out_features=self.layer_size)] +
                                      [torch.nn.Linear(in_features=self.layer_size, out_features=self.layer_size)
                                       for _ in range(self.linear_layers - 1)])
        
        self.basis_parameters = torch.nn.Linear(in_features=self.layer_size, out_features=self.theta_size)
        # self.basis_function = basis_function

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # @x: tensor of shape (N, E, S)
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)




# multivariate nbeats block only with linear layers
class MultivariateNBeatsBlock(NBeatsBlockBase):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_block()
     
    def init_block(self):

        if self.linear_layers % 2 != 0:
          raise Exception("n_layers must be divisible by 2")

        split_layers = self.linear_layers // 2
        merged_layers = self.linear_layers // 2

        branch_layer_size = self.layer_size//2

        

        # general Linear: (N,∗,H_in) --> (N,∗,H_out)
        # linear timeInputLayer: (N, E, S) --> (N, E, 512)
        self.timeInputLayer = torch.nn.Linear(in_features=self.input_size, out_features=branch_layer_size) 

        # linear featuresInputLayer: (N, S, E) --> (N, S, 512)
        self.featuresInputLayer = torch.nn.Linear(in_features=self.input_dim, out_features=branch_layer_size) 

        # linear timeLayers input: (N, E, 512) --> (N, E, S)
        self.timeLayers =  torch.nn.ModuleList(
            [torch.nn.Linear(in_features=branch_layer_size, out_features=branch_layer_size)  
            for _ in range(max(0, split_layers - 2))] 

            + [torch.nn.Linear(in_features=branch_layer_size, out_features=self.input_size)]
        )

        # linear featureLayers input: (N, S, 512) --> (N, S, E)
        self.featureLayers =  torch.nn.ModuleList(
            [torch.nn.Linear(in_features=branch_layer_size, out_features=branch_layer_size)  
            for _ in range(max(0, split_layers - 2))] 

            + [torch.nn.Linear(in_features=branch_layer_size, out_features=self.input_dim)]
        )

        # in forward:
        # transpose: featureLayers (N, S, E) --> (N, E, S)
        # concat timeLayers and featureLayers to output shape: (N, E, 2*S)
        # linear layers : (N, E, 2*S) --> (N, E, 512)
        self.layers =  torch.nn.ModuleList(
            [torch.nn.Linear(in_features=2*self.input_size, out_features=self.layer_size)] +
            [
              torch.nn.Linear(in_features=self.layer_size, out_features=self.layer_size)  
              for _ in range(max(0, merged_layers - 1))
            ] 
        )

        # basis_parameters input: (N, E, 512) --> (N, E, S+T)
        self.basis_parameters = torch.nn.Linear(in_features=self.layer_size, out_features=self.theta_size) 
        # self.basis_function = basis_function

      

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # @x: tensor of shape (N, E, S) 
        block_input = x
        
        time_block_input = torch.relu(self.timeInputLayer(block_input)) # (N, E, S) --> (N, E, 512) 

        features_block_input = torch.relu(self.featuresInputLayer(block_input.transpose(1, 2))) # (N, S, E) --> (N, S, 512)

        # (N, E, 512) --> (N, E, S) 
        for layer in self.timeLayers:
          time_block_input = torch.relu(layer(time_block_input))

        # (N, S, 512) --> (N, S, E)
        for layer in self.featureLayers:
          features_block_input = torch.relu(layer(features_block_input))

        #transpose featureLayers (N, S, E) --> (N, E, S)
        features_block_input = features_block_input.transpose(1, 2)

        # concat features_block_input and time_block_input
        block_input = torch.cat([time_block_input, features_block_input], dim=-1) # (N, E, S) --> (N, E, 2*S)

        # (N, E, 2*S) --> (N, E, 512)
        for layer in self.layers:
          block_input = torch.relu(layer(block_input))

        # (N, E, 512) --> (N, E, S+T)
        basis_parameters = self.basis_parameters(block_input) 
        return self.basis_function(basis_parameters) #outputs:  backcast: (N,E,S), forecast: (N,E,T)




# attention over time steps
class FeatureAttentionNBeatsBlock(NBeatsBlockBase):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_block()
        
    def init_block(self):

        if self.attention_embedding_size % self.attention_heads != 0:
          raise Exception("attention_embedding_size must be divisible by attention_heads")



        # input linear layer:
        # number of timeseries channels is reduced to E_small, since large numbers in first dimension cause cuda memroy errors
        # (N, S, E) --> (N, S, E_small) 
        self.inputLayer = torch.nn.Linear(in_features=self.input_dim, out_features=self.attention_embedding_size)

        # attention input: query: (E_small, N, S), key: (E_small, N, S), value: (E_small, N, S) 
        # attention output: (E_small, N, S) 
        self.selfAttentionLayers = torch.nn.ModuleList(
                        [torch.nn.MultiheadAttention(embed_dim=self.input_size, num_heads=self.attention_heads, dropout=self.dropout)  # embed_dim must be divisible by num_headsyy 
                        for _ in range(self.attention_layers)]
                  ) 
        
        # (N, S, E_small) --> (N, S, E)
        self.afterAttentionLayer = torch.nn.Linear(in_features=self.attention_embedding_size, out_features=self.input_dim)
              

        # linear layers: (N, E, S) --> (N, E, 512)
        self.layers =  torch.nn.ModuleList(
            [torch.nn.Linear(in_features=self.input_size, out_features=self.layer_size)] +
            [torch.nn.Linear(in_features=self.layer_size, out_features=self.layer_size)  
            for _ in range(max(0, self.linear_layers - 2))] # minus input layer and basis_parameters
        )

        # (N, E, 512) --> (N, E, S+T)
        self.basis_parameters = torch.nn.Linear(in_features=self.layer_size, out_features=self.theta_size) 
        # self.basis_function = basis_function

      

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # @x: tensor of shape (N, E, S) 
        block_input = x
        block_input = torch.relu(self.inputLayer(block_input.transpose(1, 2))) # (N, S, E) --> (N, S, E_small)

        block_input = block_input.transpose(0, 1).transpose(0, 2) # (N, S, E_small) -->  (E_small, N, S) 
 
        for layer in self.selfAttentionLayers: 
            # attention input: query: (E, N, S), key: (E, N, S), value: (E, N, S) 
            # attention output: (E, N, S) where E=E_small, N=bs
            block_input, _ = layer(block_input, block_input, block_input) # attention layer output (E, N, S)
            _.detach()

        # (E_small, N, S) --> (N, S, E_small)
        block_input = block_input.transpose(0, 1).transpose(1, 2)

        # # (N, S, E_small) --> (N, S, E)
        block_input = torch.relu(self.afterAttentionLayer(block_input))

        # (N, S, E) --> (N, E, S)
        block_input = block_input.transpose(1, 2) 

        # (N, E, S) --> (N, E, 512)
        for layer in self.layers:
          block_input = torch.relu(layer(block_input))

        # (N, E, 512) --> (N, E, S+T)
        basis_parameters = self.basis_parameters(block_input) 
        return self.basis_function(basis_parameters) #outputs:  backcast: (N,E,S), forecast: (N,E,T)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.2, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



# attention over time steps (standard)
class TimeAttentionNBeatsBlock(NBeatsBlockBase):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_block()
        
        
    def init_block(self):

        if self.attention_embedding_size % self.attention_heads != 0:
          raise Exception("attention_embedding_size must be divisible by attention_heads")

        # general Linear: (N,∗,H_in) --> (N,∗,H_out)
        # linear layer: (N, S, E) --> (N, S, Embedding_size)
        self.inputLayer = torch.nn.Linear(in_features=self.input_dim, out_features=self.attention_embedding_size) 

        # linear layer: (N, S, Embedding_size) --> (N, S, Embedding_size)
        if self.use_positional_encoding:
          self.positional_encoding = PositionalEncoding(d_model=self.attention_embedding_size, dropout=self.dropout)

        # attention input: query: (S, N, Embedding_size), key: (S, N, Embedding_size), value: (S, N, Embedding_size) 
        # attention output: (S, N, Embedding_size) 
        self.selfAttentionLayers = torch.nn.ModuleList(
                        [torch.nn.MultiheadAttention(embed_dim=self.attention_embedding_size, num_heads=self.attention_heads, dropout=self.dropout)  # embed_dim must be divisible by num_headsyy 
                        for _ in range(self.attention_layers)]
                  ) 

              

        # linear layers: (N, S, Embedding_size) --> (N, S, E) 
        self.layers =  torch.nn.ModuleList(
            
            [torch.nn.Linear(in_features=self.attention_embedding_size, out_features=self.attention_embedding_size)  
            for _ in range(max(0, self.linear_layers - 2))] # minus input layer and basis_parameters

            + [torch.nn.Linear(in_features=self.attention_embedding_size, out_features=self.input_dim) ]
        )

        # basis_parameters: (N, E, S) --> (N, E, S+T)
        self.basis_parameters = torch.nn.Linear(in_features=self.input_size, out_features=self.theta_size) 
        # self.basis_function = basis_function

      

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # @x: tensor of shape (N, E, S) 
        block_input = x
        block_input = torch.relu(self.inputLayer(block_input.permute(0, 2, 1))) #linear layer: (N, S, E) --> (N, S, Embedding_size)

        if self.use_positional_encoding:
          block_input = self.positional_encoding(block_input)

        block_input = block_input.permute(1, 0, 2) # shape: (S, N, Embedding_size)
 
        for layer in self.selfAttentionLayers: 
            # attention layer input: query: (S, N, Embedding_size), key: (S, N, Embedding_size), value: (S, N, Embedding_size) 
            # attention output: (S, N, Embedding_size) 
            block_input, _ = layer(block_input, block_input, block_input) # attention layer output (S, N, Embedding_size)

        block_input = block_input.permute(1, 0, 2) # (S, N, Embedding_size) --> (N, S, Embedding_size)


        # linear layers: (N, S, Embedding_size) --> (N, S, E) 
        for layer in self.layers:
          block_input = torch.relu(layer(block_input))

        # basis_parameters: (N, E, S) --> (N, E, S+T)
        basis_parameters = self.basis_parameters(block_input.permute(0, 2, 1)) # outputs: (N, E, S+T)
        return self.basis_function(basis_parameters) #outputs:  backcast: (N,E,S), forecast: (N,E,T)


