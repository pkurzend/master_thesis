
from typing import Tuple, List
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from .basis_functions import GenericBasis, TrendBasis, SeasonalityBasis, MultivariateBasis, ExogenousBasisInterpretable
from .blocks import NBeatsBlock, MultivariateNBeatsBlock, TimeAttentionNBeatsBlock, FeatureAttentionNBeatsBlock
from .blocks import SimpleNBeatsBlock, LinearNBeatsBlock, LinearAttentionNBeatsBlock, LinearTransformerEncoderNBeatsBlock, LinearConvNBeatsBlock


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NBeats(torch.nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self, output_dim,  stack_features_along_time, blocks: torch.nn.ModuleList):
        super().__init__()
        self.blocks = blocks
        self.output_dim=output_dim
        self.stack_features_along_time=stack_features_along_time
        self.first_batch = True






    
    def forward(self, x_ts : torch.Tensor, x_tf : torch.Tensor, x_s : torch.Tensor, pad_mask : torch.Tensor) -> torch.Tensor:
        # @x_ts: time series inputs, shape: (N, S, E) = (batch_size, context_length, target_dim * n_lags) 
        # @x_tf: time features inputs, shape: (N, S, n_features) = (batch_size, context_length, 4)
        # @x_s: static time feature inputs, shape: (N, S, target_dim * embed_dim)  = (batch_size, context_length, target_dim * 1) 
        # @pad_mask: padding maks, shape: (N, S) = (N, context_length)

        # @retrun: shape: (N, T, E)


        if self.stack_features_along_time:
          x =  torch.cat((x_ts, x_s), dim=1)
        else:
          x =  torch.cat((x_ts, x_s, x_tf), dim=-1) # shape: (N, context_length, input_dim)
          # x = x_ts



        # if self.first_batch:
        #   print('x_ts ', x_ts.shape)
        #   print('x_tf ', x_tf.shape)
        #   print('x_s ', x_s.shape)
        #   print('x ', x.shape)
        #   print('pad_mask ', pad_mask.shape)
        #   self.first_batch = False

        x = x.transpose(1, 2) # shape: (N, E, S)

        # input_mask = torch.ones(x.shape).to(device)
        input_mask = pad_mask.unsqueeze(1).expand(-1, x.shape[1], -1)# (N, E, S)
        

        # flip: reverse order in given axis: we want to
        # reverse time series order (last dimension) 
        residuals = x.flip(dims=(2,)) # shape: (N, E, S)
        

        forecast = x[:, :self.output_dim, -1:]
        # print('forecast (shape, min, max, mean): ', forecast.shape, forecast.min().item(), forecast.max().item(), forecast.mean().item())

        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals) # backcast: (N,E,S)
            #print(block_forecast)
            
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast[:, :self.output_dim, :]
            # print('block_forecast (shape, min, max, mean): ', block_forecast.shape, block_forecast.min().item(), block_forecast.max().item(), block_forecast.mean().item())
        return forecast.transpose(1, 2) # (N,E,T) --> (N, T, E)




class MultivariateNBeats(torch.nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self, output_dim,  stack_features_along_time, blocks: torch.nn.ModuleList):
        super().__init__()
        self.blocks = blocks
        self.output_dim=output_dim
        self.stack_features_along_time=stack_features_along_time
        self.first_batch = True






    
    def forward(self, x_ts : torch.Tensor, x_tf : torch.Tensor, x_s : torch.Tensor, pad_mask : torch.Tensor) -> torch.Tensor:
        # @x_ts: time series inputs, shape: (N, S, E) = (batch_size, context_length, target_dim * n_lags) 
        # @x_tf: time features inputs, shape: (N, S, n_features) = (batch_size, context_length, 4)
        # @x_s: static time feature inputs, shape: (N, S, target_dim * embed_dim)  = (batch_size, context_length, target_dim * 1) 
        # @pad_mask: padding maks, shape: (N, S) = (N, context_length)

        # @retrun: shape: (N, T, E)

        # print('output_dim ', self.output_dim)
        # print('x_ts.shape ', x_ts.shape)

        # x =  torch.cat((x_ts, x_s, x_tf), dim=-1) # shape: (N, context_length, input_dim)
        x = x_ts[:, :, :self.output_dim] # (N, S, E)
        # print('x.shape ', x.shape)
        N, S, E = x.shape

  
       

        # input_mask = torch.ones(x.shape).to(device)
        input_mask = pad_mask.unsqueeze(1).expand(-1, x.shape[2], -1)# (N, E, S)

        # flatten input:
        x = x.reshape(x.shape[0], -1) # (N, S*E)
        # print('x_reshaped.shape ', x.shape)

        # flatten input_mask
        input_mask = input_mask.reshape(x.shape[0], -1) # (N, S*E)
        # print('input_mask.shape ', input_mask.shape)


        # forecast = x[:, :self.output_dim, -1:]
        forecast = x[:, -1:] *0 # (N, 1) 
        # print('forecast.shape ', forecast.shape)
        

        # flip: reverse order in given axis: we want to
        # reverse time series order (last dimension) 
        residuals = x.flip(dims=(-1,)) # shape:  (N, S*E)
        # print('residuals.shape ', residuals.shape)
        



        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals) # forecast: (N, E*T), backcast: (N,E*S)
            # print('backcast and forecast shape ', backcast.shape, block_forecast.shape)
            #print(block_forecast)
            
            residuals = (residuals - backcast) * input_mask
            # forecast = forecast + block_forecast[:, :self.output_dim, :]
            forecast = forecast + block_forecast #.reshape(x.shape[0], self.output_dim, -1) # (N,E,T)
        forecast = forecast.reshape(forecast.shape[0], self.output_dim, -1)
        return forecast.transpose(1, 2) # (N,E,T) --> (N, T, E)







def generate_model(input_size: int, 
            output_size: int, 
            input_dim: int,
            output_dim = None,
            stack_features_along_time=False,
            stacks: int=30, # number of generic blocks
            interpretable: bool = False,
            multivariate_stacks : int = 0,
            linear_layers: int=4, 
            layer_size: int=512, 
            block : nn.Module = NBeatsBlock,  
            attention_layers : int=1, 
            attention_embedding_size : int=512, 
            attention_heads : int = 1,
            positional_encoding : bool = True,                                            
            dropout=0.1,
            use_dropout_layer=False,

            # parameters for interpretable verions
            degree_of_polynomial : int = 3,
            trend_layer_size : int = 256,
            seasonality_layer_size : int = 2048,
            num_of_harmonics : int = 1,

            multivariate_nbeats_like_darts : bool = False,

            ):
    """
    Create N-BEATS generic model.
    """
    print(F'NBEATS using {block} blocks')

    if multivariate_nbeats_like_darts:
        a = 2
        # input x will be of shape (N, S*D)
        # output will be of shape (N, T*D)
        # but backcast and forecast outouts are (N, *, S) nd (N, *, T)
        # output has to be of shape (N, S*D) and (N, T*D)
        # basis parameters should have layer_size (S+T)*D
        # basis function has to be changes since it returns (N, E, S) or (N, E, T)

        
        block = NBeatsBlock

        input_size = input_size * output_dim
        output_size = output_size * output_dim


    if not interpretable:
        blocks = torch.nn.ModuleList([block(   input_size=input_size,
                                                    input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    theta_size=input_size + output_size,
                                                    basis_function=GenericBasis(backcast_size=input_size,
                                                                                forecast_size=output_size),
                                                    linear_layers=linear_layers,
                                                    layer_size=layer_size,
                                                    attention_layers=attention_layers,
                                                    attention_embedding_size=attention_embedding_size,
                                                    attention_heads=attention_heads,
                                                    positional_encoding=positional_encoding,
                                                    dropout=dropout,
                                                    use_dropout_layer=use_dropout_layer
                                                  )
                                      for _ in range(stacks)]
                                    + [block(   input_size=input_size,
                                                    input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    theta_size=input_size + output_size,
                                                    basis_function=MultivariateBasis(input_dim=input_dim, backcast_size=input_size,
                                                                                forecast_size=output_size),
                                                    linear_layers=linear_layers,
                                                    layer_size=layer_size,
                                                    attention_layers=attention_layers,
                                                    attention_embedding_size=attention_embedding_size,
                                                    attention_heads=attention_heads,
                                                    positional_encoding=positional_encoding,
                                                    dropout=dropout,
                                                    use_dropout_layer=use_dropout_layer
                                                  )
                                      for _ in range(multivariate_stacks)])
    else: # interpretable
        assert stacks % 2 == 0, 'in interpretable mode, stacks must be divisible by 2 since there will be stacks // 2 trend blocks and stacks//2 seasonality blocks'
        # in interpretable version, weights within one stack are shared



        trend_block = block(input_size=input_size,
                            input_dim=input_dim,
                            output_dim=output_dim,
                            theta_size=2 * (degree_of_polynomial + 1),
                            basis_function=TrendBasis(degree_of_polynomial=degree_of_polynomial,
                                                        backcast_size=input_size,
                                                        forecast_size=output_size),
                            linear_layers=linear_layers,
                            layer_size=trend_layer_size,
                            attention_layers=attention_layers,
                            attention_embedding_size=attention_embedding_size,
                            attention_heads=attention_heads,
                            positional_encoding=positional_encoding,
                            dropout=dropout,
                            use_dropout_layer=use_dropout_layer
                          )
        
        seasonality_block = block(  input_size=input_size,
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    theta_size=4 * int(
                                        np.ceil(num_of_harmonics / 2 * output_size) - (num_of_harmonics - 1)),
                                    basis_function=SeasonalityBasis(harmonics=num_of_harmonics,
                                                                    backcast_size=input_size,
                                                                    forecast_size=output_size),
                                    linear_layers=linear_layers,
                                    layer_size=seasonality_layer_size,
                                    attention_layers=attention_layers,
                                    attention_embedding_size=attention_embedding_size,
                                    attention_heads=attention_heads,
                                    positional_encoding=positional_encoding,
                                    dropout=dropout,
                                    use_dropout_layer=use_dropout_layer
                                  )
        
        multivariate_block = block(   input_size=input_size,
                                                    input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    theta_size=input_size + output_size,
                                                    basis_function=MultivariateBasis(input_dim=input_dim, backcast_size=input_size,
                                                                                forecast_size=output_size),
                                                    linear_layers=linear_layers,
                                                    layer_size=layer_size,
                                                    attention_layers=attention_layers,
                                                    attention_embedding_size=attention_embedding_size,
                                                    attention_heads=attention_heads,
                                                    positional_encoding=positional_encoding,
                                                    dropout=dropout,
                                                    use_dropout_layer=use_dropout_layer
                                                  )

        blocks =  torch.nn.ModuleList(
            [trend_block for _ in range(stacks//2)] + [seasonality_block for _ in range(stacks//2)] + [multivariate_block for _ in range(multivariate_stacks)])

    if multivariate_nbeats_like_darts and not interpretable:
      print('using multivariate nbeats model like in DARTS library')
      model = MultivariateNBeats(output_dim=output_dim, stack_features_along_time=stack_features_along_time, blocks=blocks)
    else:
      model = NBeats(output_dim=output_dim, stack_features_along_time=stack_features_along_time, blocks=blocks)
    return model











    














def generate_model_old(input_size: int, 
            output_size: int, 
            input_dim: int,
            output_dim = None,
            stack_features_along_time=False,
            stacks: int=30, # number of generic blocks
            interpretable: bool = False,
            covariate_blocks : int = 0,
            linear_layers: int=4, 
            layer_size: int=512, 
            block : nn.Module = NBeatsBlock,  
            attention_layers : int=1, 
            attention_embedding_size : int=512, 
            attention_heads : int = 1,
            positional_encoding : bool = True,                                            
            dropout=0.1,
            use_dropout_layer=False,

            # parameters for interpretable verions
            degree_of_polynomial : int = 3,
            trend_layer_size : int = 256,
            seasonality_layer_size : int = 2048,
            num_of_harmonics : int = 1,

            ):
    """
    Create N-BEATS generic model.
    """
    print(F'NBEATS using {block} blocks')
    if not interpretable:
        blocks = torch.nn.ModuleList([block(   input_size=input_size,
                                                    input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    theta_size=input_size + output_size,
                                                    basis_function=GenericBasis(backcast_size=input_size,
                                                                                forecast_size=output_size),
                                                    linear_layers=linear_layers,
                                                    layer_size=layer_size,
                                                    attention_layers=attention_layers,
                                                    attention_embedding_size=attention_embedding_size,
                                                    attention_heads=attention_heads,
                                                    positional_encoding=positional_encoding,
                                                    dropout=dropout,
                                                    use_dropout_layer=use_dropout_layer
                                                  )
                                      for _ in range(stacks)]
                                    + [block(   input_size=input_size,
                                                    input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    theta_size=input_size + output_size,
                                                    basis_function=GenericBasis(backcast_size=input_size,
                                                                                forecast_size=output_size),
                                                    linear_layers=linear_layers,
                                                    layer_size=layer_size,
                                                    attention_layers=attention_layers,
                                                    attention_embedding_size=attention_embedding_size,
                                                    attention_heads=attention_heads,
                                                    positional_encoding=positional_encoding,
                                                    dropout=dropout,
                                                    use_dropout_layer=use_dropout_layer
                                                  )
                                      for _ in range(covariate_blocks)])
    else: # interpretable
        assert stacks % 2 == 0, 'in interpretable mode, stacks must be divisible by 2 since there will be stacks // 2 trend blocks and stacks//2 seasonality blocks'
        # in interpretable version, weights within one stack are shared



        trend_block = block(input_size=input_size,
                            input_dim=input_dim,
                            output_dim=output_dim,
                            theta_size=2 * (degree_of_polynomial + 1),
                            basis_function=TrendBasis(degree_of_polynomial=degree_of_polynomial,
                                                        backcast_size=input_size,
                                                        forecast_size=output_size),
                            linear_layers=linear_layers,
                            layer_size=trend_layer_size,
                            attention_layers=attention_layers,
                            attention_embedding_size=attention_embedding_size,
                            attention_heads=attention_heads,
                            positional_encoding=positional_encoding,
                            dropout=dropout,
                            use_dropout_layer=use_dropout_layer
                          )
        
        seasonality_block = block(  input_size=input_size,
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    theta_size=4 * int(
                                        np.ceil(num_of_harmonics / 2 * output_size) - (num_of_harmonics - 1)),
                                    basis_function=SeasonalityBasis(harmonics=num_of_harmonics,
                                                                    backcast_size=input_size,
                                                                    forecast_size=output_size),
                                    linear_layers=linear_layers,
                                    layer_size=seasonality_layer_size,
                                    attention_layers=attention_layers,
                                    attention_embedding_size=attention_embedding_size,
                                    attention_heads=attention_heads,
                                    positional_encoding=positional_encoding,
                                    dropout=dropout,
                                    use_dropout_layer=use_dropout_layer
                                  )
        
        covariates_block = block(   input_size=input_size,
                                                    input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    theta_size=input_size + output_size,
                                                    basis_function=GenericBasis(backcast_size=input_size,
                                                                                forecast_size=output_size),
                                                    linear_layers=linear_layers,
                                                    layer_size=layer_size,
                                                    attention_layers=attention_layers,
                                                    attention_embedding_size=attention_embedding_size,
                                                    attention_heads=attention_heads,
                                                    positional_encoding=positional_encoding,
                                                    dropout=dropout,
                                                    use_dropout_layer=use_dropout_layer
                                                  )

        blocks =  torch.nn.ModuleList(
            [trend_block for _ in range(stacks//2)] + [seasonality_block for _ in range(stacks//2)] + [covariates_block for _ in range(covariate_blocks)])

    return NBeats(output_dim=output_dim, stack_features_along_time=stack_features_along_time, blocks=blocks)