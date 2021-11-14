import sys
import os

plots_folder = 'plots'

# creates folders
folders = ['gridresults', 'errs', 'logs', plots_folder]
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass

    
sys.path.append('../../')

from models.nbeats import NBeatsBlock, SimpleNBeatsBlock, LinearNBeatsBlock, LinearAttentionNBeatsBlock, LinearTransformerEncoderNBeatsBlock, LinearConvNBeatsBlock
from models.estimators.nbeats_estimator import NBEATSEstimator, NBEATSFlowEstimator
from dataset.loader import load_dataset, generate_data, plot_data
import sys 

from models.estimators.trainer import Trainer

from models.utils import plot1, plot2, count_parameters, plot_forecasts



from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
# from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
from gluonts.dataset.split import OffsetSplitter
from gluonts.dataset.common import ListDataset

from gluonts.transform import (
    InstanceSplitter,
    InstanceSampler,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    UniformSplitSampler
  )

import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonts.time_feature import get_seasonality
import numpy as np
import os 
from subprocess import call

import matplotlib.pyplot as plt


print('__Number CUDA Devices:', torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

block_types = [SimpleNBeatsBlock, LinearNBeatsBlock, LinearAttentionNBeatsBlock, LinearTransformerEncoderNBeatsBlock, LinearConvNBeatsBlock]




# get hyperparameters from sys args
hp_names = [

    ('dataset_id', int),
    ('run_id', str),
    ('dataset_name', str),   

]


hp_dict = {   hp_name : hp_type(sys.argv[i+1]) for i, (hp_name, hp_type) in enumerate(hp_names)   }

dataset_id = hp_dict['dataset_id']
run_id = hp_dict['run_id']
dataset_name = hp_dict['dataset_name']


learning_rate = 1e-4
hp_dict['learning_rate'] = learning_rate

batch_size = 64
hp_dict['batch_size'] = batch_size

weight_decay = 0
hp_dict['weight_decay'] = weight_decay


blocks = NBeatsBlock
hp_dict['blocks'] = blocks


stacks = 3
hp_dict['stacks'] = stacks

layer_size = 32
hp_dict['layer_size'] = layer_size

linear_layers = 4
hp_dict['linear_layers'] = linear_layers


loss_function='sMAPE'
hp_dict['loss_function'] = loss_function




# hyperparameters not tuned
max_learning_rate = 1e-3
hp_dict['max_learning_rate'] = max_learning_rate


if batch_size==32:
    num_batches_per_epoch = 200
else:
    num_batches_per_epoch = 100
hp_dict['num_batches_per_epoch'] = num_batches_per_epoch





stack_features_along_time = 0
hp_dict['stack_features_along_time'] = stack_features_along_time

positional_encoding = 0
hp_dict['positional_encoding'] = positional_encoding

dropout = 0.5
hp_dict['dropout'] = dropout

use_dropout_layer = 0
hp_dict['use_dropout_layer'] = use_dropout_layer

dequantize = True if dataset_name in ['solar_nips', 'taxi_30min'] else False
print('dequantize: ', dequantize)
hp_dict['dequantize'] = dequantize


multivariate_nbeats_like_darts = True
hp_dict['multivariate_nbeats_like_darts'] = multivariate_nbeats_like_darts


modelpath = f'{dataset_id}_{run_id}'


print('hyperparameter dict: ')
print(hp_dict)

print('DATASET NAME: ', dataset_name)
print('BLOCK TYPE: ', blocks)

start = '''
############################################################
###################### START TRAINING ######################
############################################################
# '''
print(start)


split_offset = 0
dataset, dataset_train, dataset_test = load_dataset(name=dataset_name, validation_set=False)
# dataset, dataset_train, dataset_val, dataset_test, split_offset = load_dataset(name=dataset_name, train_pct=0.7)

evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                target_agg_funcs={'sum': np.sum})

target_dim=int(dataset.metadata.feat_static_cat[0].cardinality)
prediction_length = dataset.metadata.prediction_length
context_length = prediction_length * 4
freq = dataset.metadata.freq





estimator = NBEATSEstimator(
    target_dim=target_dim,
    prediction_length=prediction_length,
    freq=freq,
    split_offset=split_offset,

    loss_function=loss_function,
    context_length=context_length,
    stack_features_along_time=stack_features_along_time,
    block=blocks,
    stacks=stacks,
    linear_layers=linear_layers,
    layer_size=layer_size,


    positional_encoding=positional_encoding,
    dropout=dropout,
    use_dropout_layer=use_dropout_layer,

    multivariate_nbeats_like_darts = multivariate_nbeats_like_darts,
 




    trainer=Trainer(device=device,
                    epochs=25,
                    learning_rate=learning_rate,
                    maximum_learning_rate=max_learning_rate,
                    num_batches_per_epoch=num_batches_per_epoch,
                    batch_size=batch_size,
                    clip_gradient=None,
                    weight_decay=weight_decay, # 0.01 is standard value in AdamW optimizer
                    # optimizer="AdamW"
                    )
)


predictor = estimator.train(dataset_train)

number_of_parameters = count_parameters(predictor.prediction_net.nb_model)

print('NUMBER OF PARAMETERS: ', number_of_parameters)

train_losses = estimator.history['train_epoch_losses']
val_losses = estimator.history['val_epoch_losses']
print('len train_losses ', len(train_losses))
print('len val_losses ', len(val_losses))

nan_loss = False
for train_loss in train_losses:
    
    if  np.isnan(train_loss):
        nan_loss = True





forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                            predictor=predictor,
                                            num_samples=1)


forecasts = list(forecast_it)
targets = list(ts_it)

if len(targets)<len(forecasts):
  targets =targets*len(forecasts)


agg_metric, _ = evaluator(targets, forecasts, num_series=len(targets))


plot_forecasts(targets, forecasts, modelpath)

mse = agg_metric['MSE']
mase = agg_metric['MASE']
mape = agg_metric['MAPE']
smape = agg_metric['sMAPE']
crps = agg_metric['mean_wQuantileLoss']

mse_sum = agg_metric['m_sum_MSE']
mase_sum = agg_metric['m_sum_MASE']
mape_sum = agg_metric['m_sum_MAPE']
smape_sum = agg_metric['m_sum_sMAPE']
crps_sum = agg_metric['m_sum_mean_wQuantileLoss']


print("mse: {}".format(mse))
print("mase: {}".format(mase))
print("mape: {}".format(mape))
print("smape: {}".format(smape))
print("crps: {}".format(crps))
print("mse_sum: {}".format(mse_sum))
print("mase_sum: {}".format(mase_sum))
print("mape_sum: {}".format(mape_sum))
print("smape_sum: {}".format(smape_sum))
print("crps_sum: {}".format(crps_sum))




result_obj = {
    'metrics_val' : {
        'mse' : mse,
        'mase' : mase,
        'mape' : mape,
        'smape' : smape,
        'crps' : crps,
        'mse_sum' : mse_sum,
        'mase_sum' : mase_sum,
        'mape_sum' : mape_sum,
        'smape_sum' : smape_sum,
        'crps_sum' : crps_sum,
    },
    'nan_loss' : nan_loss,
    'losses' : estimator.history,
    'hyperparameters' : hp_dict,
    'dataset_name' : dataset_name,
    'number_of_parameters' : number_of_parameters,
    'run_id' : run_id,
    'dataset_id' : dataset_id,

 }

print(result_obj)

import pickle 
filename = F"gridresults/{modelpath}.pkl"
with open(filename, 'wb') as f:
  pickle.dump(result_obj, f)




