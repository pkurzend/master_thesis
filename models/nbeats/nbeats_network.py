# n_beats_network.py
from pts.modules import MeanScaler
from typing import List, Optional, Tuple
from typing import Tuple
import math

from typing import Tuple, List
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from gluonts.time_feature import get_seasonality

from .nbeats import generate_model 
from .blocks import NBeatsBlock 

import pytorch_model_summary as pms
from ..utils import count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys 


class NBEATSTrainingNetwork(nn.Module):
    """
    model to train nbeats network
    forward method returns loss of nbeats forecasts

    @loss_function: String, one of ["MASE", "MAPE", "sMAPE"], determins which loss function is applied in the forward method
    @input_dimension: Int, time series dimension (number of features/channels/input time series)
    @context_length: Int, input time series length
    @prediciton_length: Int, forcast horizon (length of forecast)
    @freq: Str one of (H, 1H, 10min, 1B)


    """

    def __init__(self, loss_function, input_dim, target_dim, context_length , prediction_length, freq, lags_seq, history_length,
                 stack_features_along_time=False,
                  stacks: int=30, 
                  linear_layers: int=4, 
                  layer_size: int=512, 
                  block : nn.Module = NBeatsBlock,  
                  attention_layers : int=1, 
                  attention_embedding_size : int=512, 
                  attention_heads : int = 1,
                  positional_encoding : bool = True,
                  dropout = 0.1,
                  use_dropout_layer = False,
                 
                 
                  # parameters for interpretable verions
                  interpretable : bool = False,
                  multivariate_stacks : int = 0,
                  degree_of_polynomial : int = 3,
                  trend_layer_size : int = 256,
                  seasonality_layer_size : int = 2048,
                  num_of_harmonics : int = 1,
                 
            ):
      
        super().__init__()

        
        self.loss_function = loss_function
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.freq = freq
        self.history_length = history_length
        self.stack_features_along_time=stack_features_along_time

        self.positional_encoding=positional_encoding
        self.dropout = dropout
        self.use_dropout_layer = use_dropout_layer
        self.interpretable = interpretable
        self.multivariate_stacks = multivariate_stacks
        self.degree_of_polynomial=degree_of_polynomial
        self.trend_layer_size=trend_layer_size
        self.seasonality_layer_size=seasonality_layer_size 
        self.num_of_harmonics=num_of_harmonics 

        


        self.periodicity = get_seasonality(self.freq)
        if self.periodicity > self.context_length + self.prediction_length:
            self.periodicity = 1 # no period within considered time window

        # saling of data
        self.scaler = MeanScaler(keepdim=True)


        self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )

        self.lags_seq = lags_seq

        self.firstBatchIndicator = True

        self.model_input_length = self.context_length
        if self.stack_features_along_time:
          self.model_input_length = self.context_length * len(self.lags_seq) +  self.embed_dim 

        print('model_input_length ', self.model_input_length)

        self.nb_model = generate_model(input_size=self.model_input_length , 
                                           output_size=prediction_length, 
                                           input_dim=input_dim, 
                                           output_dim=target_dim,
                                           stack_features_along_time=stack_features_along_time,
                                           block=block, 
                                           stacks=stacks,
                                           linear_layers=linear_layers,
                                           layer_size=layer_size,
                                           attention_layers=attention_layers,
                                           attention_embedding_size=attention_embedding_size,
                                           attention_heads=attention_heads,
                                           positional_encoding=positional_encoding,
                                           dropout=dropout,
                                           use_dropout_layer=use_dropout_layer,
                                           interpretable=interpretable,
                                           multivariate_stacks=self.multivariate_stacks,
                                           degree_of_polynomial=degree_of_polynomial,
                                           trend_layer_size=trend_layer_size,
                                           seasonality_layer_size=seasonality_layer_size,
                                           num_of_harmonics=num_of_harmonics
                                           )

        self.number_of_parameters = count_parameters(self.nb_model)


    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length
            length of sequence in the T (time) dimension (axis = 1).
        indices
            list of lag indices to be used.
        subsequences_length
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices),
            containing lagged subsequences. First lagged subsequence with lag=1 is the time series of the context windwo directly before the prediction window
            Specifically, lagged[i, :, j, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: history_length + begin_index >= 0
        # that is: history_length - lag_index - sequence_length >= 0
        # hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)






    def create_network_input(
        self,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target: Optional[torch.Tensor],
        target_dimension_indicator: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        past_time_feat
            Past time features (batch_size, history_length, num_features)
        past_target
            Past target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target
            Future target values (batch_size,
            prediction_length, target_dim)
        target_dimension_indicator
            Dimensionality of the time series (batch_size, target_dim)
        Returns
        -------
        inputs:
          containing time series (past_target, future_target), scaled lagged sequences, time features and index_embeddings
          (batch_size, sub_seq_len, input_dim)
        past_is_pad:
            indicators if timestep is padded, 1 where padded, 0 else
            (batch_size, history_length)
        scale:
            scale computed on context_length
            (batch_size, 1, target_dim)

        index_embeddings:
            index embeddings
            (batch_size, target_dim, embed_dim)
        

        """

    

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        if future_time_feat is None or future_target is None:
            time_feat = past_time_feat[:, -self.context_length :, ...] # shape : (N, context_length, n-features)
            sequence = past_target
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (
                    past_time_feat[:, -self.context_length :, ...],
                    future_time_feat,
                ),
                dim=1,
            )
            sequence = torch.cat((past_target, future_target), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # (batch_size, sub_seq_len, target_dim, num_lags) =  (batch_size, context_length, target_dim, num_lags)
        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )
        # print(lags.shape)
  

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, target_dim)
        _, scale = self.scaler(
            past_target[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        )

        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1)
        # print(lags_scaled.shape)
       

        # (batch_size, target_dim, embed_dim)
        index_embeddings = self.embed(target_dimension_indicator)


        if self.stack_features_along_time:
            #  (batch_size, sub_seq_len*num_lags, target_dim)
            input_lags = lags_scaled.reshape(
                (-1, subsequences_length * len(self.lags_seq), self.target_dim)
            )

            # (batch_size, subsequences_length * embed_dim, target_dim )
            repeated_index_embeddings = (
                index_embeddings.unsqueeze(1)
                # .expand(-1, subsequences_length, -1, -1)
                .reshape((-1, 1, self.target_dim)) # .reshape((-1, subsequences_length * self.embed_dim, self.target_dim))
            )
            past_is_pad = past_is_pad[:, -self.context_length:].tile(1, len(self.lags_seq)) # (N, context_length*lags)
            past_is_pad = torch.cat([past_is_pad, torch.zeros(past_is_pad.shape[0],1).to(device)], dim=1) # (N, context_length*lags + 1)
            assert past_is_pad.shape[1] == self.context_length * len(self.lags_seq) + self.embed_dim, F'has shape {past_is_pad.shape}'


            # # (batch_size, subsequences_length, target_dim * embed_dim)
            # repeated_index_embeddings = (
            #     index_embeddings.unsqueeze(1)
            #     .expand(-1, subsequences_length* len(self.lags_seq), -1, -1)
            #     .reshape((-1, subsequences_length* len(self.lags_seq), self.target_dim * self.embed_dim))
            # )

            # (batch_size, sub_seq_len) 
            # past_is_pad = past_is_pad[:, -self.context_length:].expand(-1, self.embed_dim + len(self.lags_seq))
            # past_is_pad = past_is_pad[:, -self.context_length:].tile(1, self.embed_dim + len(self.lags_seq))
            # assert past_is_pad.shape[1] == self.context_length * (self.embed_dim + len(self.lags_seq))
            

        else:
            #  (batch_size, sub_seq_len, target_dim * num_lags)
            input_lags = lags_scaled.reshape(
                (-1, subsequences_length, len(self.lags_seq) * self.target_dim)
            )

            # (batch_size, subsequences_length, target_dim * embed_dim)
            repeated_index_embeddings = (
                index_embeddings.unsqueeze(1)
                .expand(-1, subsequences_length, -1, -1)
                .reshape((-1, subsequences_length, self.target_dim * self.embed_dim))
            )

            # (batch_size, sub_seq_len) 
            past_is_pad = past_is_pad[:, -self.context_length:]

       
        


        



        # (batch_size, sub_seq_len, target_dim * n_lags) 
        # time series, also contains lagged timeseries
        time_series_inputs = input_lags

        # (batch_size, sub_seq_len, n_features) =  (batch_size, sub_seq_len, 4)
        # time dependend features
        time_feat_inputs = time_feat

        # (batch_size, sub_seq_len, target_dim * embed_dim) = (batch_size, sub_seq_len, target_dim * 1) 
        # static time features
        static_inputs = repeated_index_embeddings



        # (batch_size, sub_seq_len, input_dim)  where input_dim = timeseries_dim, lags, index_embedding_dim = 1, time_feat_dim = 4
        # inputs = torch.cat((input_lags, repeated_index_embeddings, time_feat), dim=-1)

        
        

        return time_series_inputs, time_feat_inputs, static_inputs, past_is_pad, scale





    def smape_loss(
        self, forecast: torch.Tensor, future_target: torch.Tensor
    ) -> torch.Tensor:
        denominator = (torch.abs(future_target) + torch.abs(forecast)).detach()
        flag = denominator == 0

        loss = (200 / self.prediction_length) * torch.mean(
            (torch.abs(future_target - forecast) * torch.logical_not(flag))
            / (denominator + flag),
            dim=1,
        )
        return loss

    def mape_loss(
        self, forecast: torch.Tensor, future_target: torch.Tensor
    ) -> torch.Tensor:
        denominator = torch.abs(future_target)
        flag = denominator == 0

        loss = (100 / self.prediction_length) * torch.mean(
            (torch.abs(future_target - forecast) * torch.logical_not(flag))
            / (denominator + flag),
            dim=1,
        )

        return loss




    def mase_loss(
          self,   
          forecast: torch.Tensor,
          future_target: torch.Tensor,
          past_target: torch.Tensor,
          periodicity: int = 1
          
      ) -> torch.Tensor:
          factor = 1 / (self.context_length + self.prediction_length - periodicity)

          whole_target = torch.cat((past_target, future_target), dim=1)
          seasonal_error = factor * torch.mean(
              torch.abs(
                  whole_target[:, periodicity:, ...]
                  - whole_target[:, :-periodicity:, ...]
              ),
              dim=1,
          )
          flag = seasonal_error == 0

          loss = (
              torch.mean(torch.abs(future_target - forecast), dim=1)
              * torch.logical_not(flag)
          ) / (seasonal_error + flag)

          return loss



    def forward(
        self, 
        past_target: torch.Tensor, 
        future_target: torch.Tensor, 
        past_is_pad: torch.Tensor,
        past_time_feat : torch.Tensor, 
        future_time_feat : torch.Tensor,
        past_observed_values: torch.Tensor,
        future_observed_values: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
    ) -> torch.Tensor:
        # @ past_target: shape: (N, S, E)
        # @ future_target: shape: (N, T, E)
        # @ past_target: shape: (N, S, E),  torch.Size([32, 288, 963])
        # @ future_target: shape: (N, T, E),  torch.Size([32, 24, 963])
        # @ past_is_pad: shape: (N, S),   torch.Size([32, 288])
        # @ past_time_feat: shape: (N, S, n_features),   torch.Size([32, 288, 4])
        # @ future_time_feat: shape: (N, T, n_features),   torch.Size([32, 24, 4])
        # @ past_observed_values: shape: (N, S, E),   torch.Size([32, 288, 963])
        # @ future_observed_values: shape: (N, T, E),   torch.Size([32, 24, 963])
        # @ target_dimension_indicator: shape: (N, E),   torch.Size([32, 963])


        # also scales the lagged sequences which also contains the actual sequence
        time_series_inputs, time_feat_inputs, static_inputs, past_is_pad, scale = self.create_network_input(
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target=None,
            target_dimension_indicator=target_dimension_indicator,
        )

        time_series_inputs = time_series_inputs[:, : self.model_input_length, ...] # does nothing when future_target and future_time_feat are None
        time_feat_inputs = time_feat_inputs[:, : self.model_input_length, ...]
        static_inputs = static_inputs[:, : self.model_input_length, ...]

        # invert padding mask such that it is 0 where padded and 1 where not padded
        ones = torch.ones(past_is_pad.shape).to(device)
        zeros = torch.zeros(past_is_pad.shape).to(device)
        pad_mask = torch.where(past_is_pad == 1, zeros, ones)


        # forecast
        forecast = self.nb_model.forward(x_ts=time_series_inputs, x_tf=time_feat_inputs, x_s=static_inputs, pad_mask=pad_mask) #shape: (N, T, E)


  






        # future_target = future_target / scale


        # apply loss function
        if self.loss_function == "sMAPE":
            loss = self.smape_loss(forecast, future_target)
        elif self.loss_function == "MAPE":
            loss = self.mape_loss(forecast, future_target)
        elif self.loss_function == "MASE":
            loss = self.mase_loss(
                forecast, future_target, past_target / scale, self.periodicity
            )
        elif self.loss_function == "MSE":
            loss = torch.nn.MSELoss()(forecast, future_target)
        else:
            raise ValueError(
                f"Invalid value {self.loss_function} for argument loss_function."
            )


        # print('loss: ', 'min: ', loss.min().item(), 'max: ', loss.max().item(), loss.shape, 'forecast: ', 'min: ', forecast.min().item(), 'max: ', forecast.max().item(), 'mean: ', forecast.mean().item(), 'difference: ', (forecast-future_target).min().item() ,(forecast-future_target).max().item(),  'forecast nans? ', torch.isnan(forecast).any().item())

        return loss.mean()



class NBEATSPredictionNetwork(NBEATSTrainingNetwork):
    def __init__(self, **kwargs):
                 
        super().__init__(**kwargs)



    def forward(
        self, 
        past_target: torch.Tensor, 
        future_target: torch.Tensor, 
        past_is_pad: torch.Tensor,
        past_time_feat : torch.Tensor, 
        future_time_feat : torch.Tensor,
        past_observed_values: torch.Tensor,
        future_observed_values: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
    ) -> torch.Tensor:
        # @ past_target: shape: (N, S, E)
        # @ future_target: shape: (N, T, E)
        # @ past_target: shape: (N, S, E),  torch.Size([32, 288, 963])
        # @ future_target: shape: (N, T, E),  torch.Size([32, 24, 963])
        # @ past_is_pad: shape: (N, S),   torch.Size([32, 288])
        # @ past_time_feat: shape: (N, S, n_features),   torch.Size([32, 288, 4])
        # @ future_time_feat: shape: (N, T, n_features),   torch.Size([32, 24, 4])
        # @ past_observed_values: shape: (N, S, E),   torch.Size([32, 288, 963])
        # @ future_observed_values: shape: (N, T, E),   torch.Size([32, 24, 963])
        # @ target_dimension_indicator: shape: (N, E),   torch.Size([32, 963])


                # also scales the lagged sequences which also contains the actual sequence
        time_series_inputs, time_feat_inputs, static_inputs, past_is_pad, scale = self.create_network_input(
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target=None,
            target_dimension_indicator=target_dimension_indicator,
        )

        time_series_inputs = time_series_inputs[:, : self.model_input_length, ...] # does nothing when future_target and future_time_feat are None
        time_feat_inputs = time_feat_inputs[:, : self.model_input_length, ...]
        static_inputs = static_inputs[:, : self.model_input_length, ...]

        # invert padding mask such that it is 0 where padded and 1 where not padded
        ones = torch.ones(past_is_pad.shape).to(device)
        zeros = torch.zeros(past_is_pad.shape).to(device)
        pad_mask = torch.where(past_is_pad == 1, zeros, ones)


        # forecast
        forecast = self.nb_model.forward(x_ts=time_series_inputs, x_tf=time_feat_inputs, x_s=static_inputs, pad_mask=pad_mask) #shape: (N, T, E)
        
        return forecast.unsqueeze(1)













