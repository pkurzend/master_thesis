from typing import Tuple, List
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GenericBasis(torch.nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor):
       # @theta: shape (N, E, theta_size) = (N, E, S+T)
       # @return: backcast: (N, E, S), forecast: (N, E, T)

       backcast, forecast = theta[..., :self.backcast_size], theta[..., -self.forecast_size:]

       return backcast, forecast




class TrendBasis(torch.nn.Module):
    """
    Polynomial function to model trend.
    """
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        # backcast_size: = S
        # forecast_sizr: = T
        # theta_size = 8
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = torch.nn.Parameter(
            torch.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=torch.float32),
            requires_grad=False)
        self.forecast_time = torch.nn.Parameter(
            torch.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=torch.float32), requires_grad=False)
        
        print('TrendBasis polynomial_size ', self.polynomial_size)
        print('TrendBasis backcast_time ', self.backcast_time.shape)
        print('TrendBasis forecast_time ', self.forecast_time.shape)
        # TrendBasis backcast_time  torch.Size([4, 120])
        # TrendBasis forecast_time  torch.Size([4, 24])


    def forward(self, theta: torch.Tensor):
        # @theta: shape (N, E, theta_size) = (N, E, S+T)
        # @return: backcast: (N, E, S), forecast: (N, E, T)
        # print('theta ', theta.shape, theta[:, :, self.polynomial_size:].shape, ) # torch.Size([64, 3856, 8]) torch.Size([64, 3856, 4]) = (N, E, n_basis_params)
        backcast = torch.einsum('bep,pt->bet', theta[:, :, self.polynomial_size:], self.backcast_time)
        forecast = torch.einsum('bep,pt->bet', theta[:, :, :self.polynomial_size], self.forecast_time)
        # print('trend basis ', backcast.shape, forecast.shape) # torch.Size([64, 3856, 120]) torch.Size([64, 3856, 24]) = (N, E, S) and (N, E, T)
        return backcast, forecast


class SeasonalityBasis(torch.nn.Module):
    """
    Harmonic functions to model seasonality.
    """
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        # backcast_size: = S
        # forecast_sizr: = T
        # theta_size = 12
        super().__init__()
        
        # n_harmonicsx1
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = torch.nn.Parameter(torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32),
                                                    requires_grad=False)# (Theta_size, S)
        
        self.backcast_sin_template = torch.nn.Parameter(torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32),
                                                    requires_grad=False) # (Theta_size, S)
        
        self.forecast_cos_template = torch.nn.Parameter(torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32),
                                                    requires_grad=False) # (Theta_size, T)
        
        self.forecast_sin_template = torch.nn.Parameter(torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32),
                                                    requires_grad=False) # (Theta_size, T)
        print('SeasonalityBasis backcast_time ', self.backcast_cos_template.shape,  self.backcast_sin_template.shape)
        print('SeasonalityBasis forecast_time ', self.forecast_cos_template.shape,  self.forecast_sin_template.shape)
        # SeasonalityBasis backcast_time  torch.Size([12, 120]) torch.Size([12, 120])
        # SeasonalityBasis forecast_time  torch.Size([12, 24]) torch.Size([12, 24])


    def forward(self, theta: torch.Tensor):
        # @theta: shape (N, E, theta_size) 
        # @return: backcast: (N, E, S), forecast: (N, E, T)
        params_per_harmonic = theta.shape[-1] // 4 # 

        backcast_harmonics_cos = torch.einsum('bep,pt->bet', theta[:, :, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        
        backcast_harmonics_sin = torch.einsum('bep,pt->bet', theta[:, :, 3 * params_per_harmonic:], self.backcast_sin_template)

        backcast = backcast_harmonics_sin + backcast_harmonics_cos

        forecast_harmonics_cos = torch.einsum('bep,pt->bet', theta[:, :, :params_per_harmonic], self.forecast_cos_template)
                                          
        forecast_harmonics_sin = torch.einsum('bep,pt->bet', theta[:, :, params_per_harmonic:2 * params_per_harmonic],self.forecast_sin_template)
                                          
        forecast = forecast_harmonics_sin + forecast_harmonics_cos
        # print('seasonality basis: ', backcast.shape, forecast.shape)

        return backcast, forecast



class MultivariateBasis(torch.nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, input_dim,  backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.input_dim = input_dim


        # (N, S+T, E) --> (N, S+T, E)
        self.basis_layer = torch.nn.Linear(in_features=self.input_dim, out_features=self.input_dim) 

    def forward(self, theta: torch.Tensor):
       # @theta: shape (N, E, theta_size) = (N, E, S+T)
       # @return: backcast: (N, E, S), forecast: (N, E, T)

       x = theta
       x = self.basis_layer(x.permute(0, 2, 1))
       x = x.permute(0, 2, 1)
       backcast, forecast = x[:, :, :self.backcast_size], x[:, :, -self.forecast_size:]

       return backcast, forecast




import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations














class ExogenousBasisInterpretable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, theta: torch.Tensor, insample_x_t: torch.Tensor, outsample_x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast_basis = insample_x_t
        forecast_basis = outsample_x_t

        cut_point = forecast_basis.shape[1]
        backcast = torch.einsum('bp,bpt->bt', theta[:, :, cut_point:], backcast_basis)
        forecast = torch.einsum('bp,bpt->bt', theta[:, :, :cut_point], forecast_basis)
        return backcast, forecast


        # insample_y = windows[:, self.t_cols.index('y'), :self.input_size]
        # insample_x = windows[:, (self.t_cols.index('y')+1):self.t_cols.index('insample_mask'), :self.input_size]
        # insample_mask = windows[:, self.t_cols.index('insample_mask'), :self.input_size]

        # outsample_y = windows[:, self.t_cols.index('y'), self.input_size:]
        # outsample_x = windows[:, (self.t_cols.index('y')+1):self.t_cols.index('insample_mask'), self.input_size:]
        # outsample_mask = windows[:, self.t_cols.index('outsample_mask'), self.input_size:]
