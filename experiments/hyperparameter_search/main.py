
import sys
import os
sys.path.append('../../')


from models.nbeats import NBeatsBlock, MultivariateNBeatsBlock, TimeAttentionNBeatsBlock, FeatureAttentionNBeatsBlock
from models.estimators.nbeats_estimator import NBEATSEstimator
from dataset.loader import load_dataset, generate_data, plot_data
import sys 

from models.estimators.trainer import Trainer


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

import matplotlib.pyplot as plt


print('__Number CUDA Devices:', torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))


plots_folder = 'plots'

# creates folders
folders = ['gridresults', 'errs', 'logs', plots_folder]
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass


# get hyperparameters from sys args
hp_names = [
    ('dataset_name', str),
    ('learning_rate', float),
    ('max_learning_rate', float),
    ('batch_size', int),
    ('loss_function', str),

    ('interpretable', int),
    ('stack_features_along_time', int),
    ('block', int),
    
    
    ('stacks', int),
    ('linear_layers', int),
    ('layer_size', int),
    
    ('attention_layers', int),
    ('attention_heads', int),
    ('attention_embedding_size', int),

    
    
    # ('trend_layer_size', int),
    # ('seasonality_layer_size', int),
    # ('degree_of_polynomial', int),
]


hp_dict = {   hp_name : hp_type(sys.argv[i+1]) for i, (hp_name, hp_type) in enumerate(hp_names)   }

dataset_name = hp_dict['dataset_name']
learning_rate = hp_dict['learning_rate']
max_learning_rate = hp_dict['max_learning_rate']
batch_size = hp_dict['batch_size']
loss_function = hp_dict['loss_function']
 

interpretable = hp_dict['interpretable'] # 0 or 1
stack_features_along_time = hp_dict['stack_features_along_time'] # 0 or 1
block_types = [NBeatsBlock, MultivariateNBeatsBlock, TimeAttentionNBeatsBlock, FeatureAttentionNBeatsBlock]
block = block_types[hp_dict['block']] 

stacks = hp_dict['stacks']
linear_layers = hp_dict['linear_layers']
layer_size = hp_dict['layer_size']

attention_layers = hp_dict['attention_layers']
attention_heads = hp_dict['attention_heads']
attention_embedding_size = hp_dict['attention_embedding_size']


# trend_layer_size = hp_dict['trend_layer_size']
# seasonality_layer_size = hp_dict['seasonality_layer_size']
# degree_of_polynomial = hp_dict['degree_of_polynomial']


print('hyperparameter dict: ')
print(hp_dict)

print('DATASET NAME: ', dataset_name)
print('BLOCK TYPE: ', block)

start = '''
############################################################
###################### START TRAINING ######################
############################################################
# '''
print(start)



dataset, dataset_train, dataset_val, dataset_test, split_offset = load_dataset(name=dataset_name, train_pct=0.7)

evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                target_agg_funcs={'sum': np.sum})

target_dim=int(dataset.metadata.feat_static_cat[0].cardinality)
prediction_length = dataset.metadata.prediction_length
context_length = prediction_length * 5
freq = dataset.metadata.freq

modelpath = '&'.join([F"{param}={value}" for param, value in hp_dict.items() if param not in ['stack_features_along_time', 'attention_embedding_size', 'attention_heads', 'attention_layers']])




def plot1(targets, forecasts, modelpath):
    ts_entry = targets[0]  #<-this line is needed
    forecast_entry = forecasts[0]  #<-this line is needed


    fig, axes = plt.subplots(10, 4, figsize=(20,30))

    axes = [ax for row in axes for ax in row]

    prediction_length = dataset.metadata.prediction_length
    target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)

    for j in range(min(target_dim, 20)):
        # ts_entry[i][-120:].plot()
        # forecast_entry.copy_dim(i).plot(color='g')
        # print(ts_entry[j][-120:])
        # print(ts_entry[j][-120:].shape)
        # print(forecast_entry.copy_dim(j).samples[0])
        # print(forecast_entry.copy_dim(j).samples[0].shape)

        ground_truth_x = np.arange(120)
        model_x = np.arange(120-prediction_length, 120)

        l1, = axes[j].plot(ground_truth_x, ts_entry[j][-120:])
        l2, = axes[j].plot(model_x, forecast_entry.copy_dim(j).samples[0], color='g')
    plt.savefig(F'{plots_folder}/{modelpath}_1.png')

def plot2(targets, forecasts, modelpath):
    ts_entry = targets[0]  #<-this line is needed
    forecast_entry = forecasts[0]  #<-this line is needed
    for i in range(4):
        plt.subplot(2,2,i+1)
        ts_entry[i][-120:].plot()
        forecast_entry.copy_dim(i).plot(color='g')
    plt.savefig(F'{plots_folder}/{modelpath}_2.png')


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
    interpretable=False,
    # use_time_features=True,
    # degree_of_polynomial  = degree_of_polynomial,
    # trend_layer_size = trend_layer_size,
    # seasonality_layer_size = seasonality_layer_size,
    # num_of_harmonics  = 1,

    trainer=Trainer(device=device,
                    epochs=40,
                    learning_rate=learning_rate,
                    maximum_learning_rate=max_learning_rate,
                    num_batches_per_epoch=100,
                    batch_size=batch_size,
                    clip_gradient=3.0
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

plot1(targets, forecasts, modelpath)
plot2(targets, forecasts, modelpath)


agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))


mse = agg_metric['MSE']
mase = agg_metric['MASE']
mape = agg_metric['MAPE']
smape = agg_metric['sMAPE']

mse_sum = agg_metric['m_sum_MSE']
mase_sum = agg_metric['m_sum_MASE']
mape_sum = agg_metric['m_sum_MAPE']
smape_sum = agg_metric['m_sum_sMAPE']


print("mse: {}".format(mse))
print("mase: {}".format(mase))
print("mape: {}".format(mape))
print("smape: {}".format(smape))
print("mse_sum: {}".format(mse_sum))
print("mase_sum: {}".format(mase_sum))
print("mape_sum: {}".format(mape_sum))
print("smape_sum: {}".format(smape_sum))



result_obj = {
    'metrics' : {
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
    'dataset_name' : dataset_name,
}



import pickle 
filename = F"gridresults/{modelpath}.pkl"
with open(filename, 'wb') as f:
  pickle.dump(result_obj, f)



