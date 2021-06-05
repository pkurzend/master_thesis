

from models.nbeats import NBeatsBlock, MultivariateNBeatsBlock, TimeAttentionNBeatsBlock, FeatureAttentionNBeatsBlock
from models.estimators.nbeats_estimator import NBEATSEstimator
from dataset.loader import load_dataset, generate_data, plot_data
import sys 

from ...models.estimators.trainer import Trainer


from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
# from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
from gluonts.dataset.split import OffsetSplitter
from gluonts.dataset.common import ListDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonts.time_feature import get_seasonality
import numpy as np
import os 
from subprocess import call


print('__Number CUDA Devices:', torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))


# creates folders
folders = ['gridresults', 'errs', 'logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass


# get hyperparameters from sys args
hp_names = [
    ('stack_features_along_time', int),
    ('loss_function', str),
    ('block', int),
    ('stacks', int),
    ('linear_layers', int),
    ('layer_size', int),
    ('interpretable', int),
    ('attention_layers', int),
    ('attention_embedding_size', int),
    ('attention_heads', int),
    ('learning_rate', float),
    ('batch_size', int),
    ('trend_layer_size', int),
    ('seasonality_layer_size', int),
    ('degree_of_polynomial', int),
    ('dataset_name', str),
]
hp_dict = {   hp_names[0] : hp_names[1](sys.argv[i+1]) for i in range(len(hp_names))    }
 
stack_features_along_time = hp_dict['stack_features_along_time'] # 0 or 1
loss_function = hp_dict['loss_function']
stacks = hp_dict['stacks']
linear_layers = hp_dict['linear_layers']
layer_size = hp_dict['layer_size']
interpretable = hp_dict['interpretable'] # 0 or 1
attention_layers = hp_dict['attention_layers']
attention_embedding_size = hp_dict['attention_embedding_size']
attention_heads = hp_dict['attention_heads']
learning_rate = hp_dict['learning_rate']
batch_size = hp_dict['batch_size']
trend_layer_size = hp_dict['trend_layer_size']
seasonality_layer_size = hp_dict['seasonality_layer_size']
degree_of_polynomial = hp_dict['degree_of_polynomial']
dataset_name = hp_dict['dataset_name']

block_types = [NBeatsBlock, MultivariateNBeatsBlock, TimeAttentionNBeatsBlock, FeatureAttentionNBeatsBlock]
block = block_types[hp_dict['block']] 





dataset, dataset_train, dataset_val, dataset_test, split_offset = load_dataset(name=dataset_name, train_pct=0.7)

evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})

target_dim=int(dataset.metadata.feat_static_cat[0].cardinality)
prediction_length = dataset.metadata.prediction_length
context_length = prediction_length * 5
freq = dataset.metadata.freq

# TimeAttentionNBeatsBlock, FeatureAttentionNBeatsBlock, MultivariateNBeatsBlock, NBeatsBlock
estimator = NBEATSEstimator(
    target_dim=target_dim,
    prediction_length=prediction_length,
    freq=freq,
    split_offset=split_offset,
    loss_function=loss_function,
    context_length=context_length,
    stack_features_along_time=stack_features_along_time,
    block=block,
    stacks=stacks,
    linear_layers=linear_layers,
    layer_size=layer_size,
    attention_layers=attention_layers,
    attention_embedding_size=attention_embedding_size,
    attention_heads=attention_heads,
    interpretable=interpretable,
    # use_time_features=True,
    degree_of_polynomial  = degree_of_polynomial,
    trend_layer_size = trend_layer_size,
    seasonality_layer_size = seasonality_layer_size,
    num_of_harmonics  = 1,

    trainer=Trainer(device=device,
                    epochs=20,
                    learning_rate=learning_rate,
                    maximum_learning_rate=1e-3,
                    num_batches_per_epoch=100,
                    batch_size=batch_size,
                    clip_gradient=1.0
                    )
)


predictor = estimator.train(dataset_train, dataset_val)

train_losses = estimator.history['train_epoch_losses']
val_losses = estimator.history['val_epoch_losses']
for train_loss, val_loss in zip(train_losses, val_losses):
    print('Train loss: ', train_loss, '\t', 'Val loss: ', val_loss)



forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                             predictor=predictor,
                                             num_samples=1)


forecasts = list(forecast_it)
targets = list(ts_it)

print(len(forecasts))
print(len(targets))


agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))


mse = agg_metric['MSE']
mase = agg_metric['MASE']
mape = agg_metric['MAPE']
smape = agg_metric['sMAPE']

mse_sum = agg_metric['m_sum_MSE']
mase_sum = agg_metric['m_sum_MASE']
mape_sum = agg_metric['m_sum_MAPE']
smape_sum = agg_metric['m_sum_sMAPE']


result_obj = {
    'metrcs' : {
        'mse' : mse,
        'mase' : mase,
        'mape' : mape,
        'smape' : smape,
        'm_sum_mse' : mse_sum,
        'm_sum_mase' : mase_sum,
        'm_sum_mape' : mape_sum,
        'm_sum_smape' : smape_sum
    },
    'losses' : estimator.history,
    'hyperparameters' : hp_dict,
}


modelpath = '&'.join([F"{param}={value}" for param, value in hp_dict.items()])

import pickle 
filename = F"gridresults/{modelpath}.pkl"
with open(filename, 'wb') as f:
  pickle.dump(result_obj, f)



print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
print("ND: {}".format(agg_metric['ND']))
print("NRMSE: {}".format(agg_metric['NRMSE']))
print("MSE: {}".format(agg_metric['MSE']))

print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))