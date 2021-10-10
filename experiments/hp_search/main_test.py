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

from models.utils import plot1, plot2, count_parameters


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

block_types = [SimpleNBeatsBlock, LinearNBeatsBlock, LinearAttentionNBeatsBlock, LinearTransformerEncoderNBeatsBlock, LinearConvNBeatsBlock]




# get hyperparameters from sys args
hp_names = [

    ('run_id', int),
    ('dataset_name', str),
    ('learning_rate', float),
    
    
    ('blocks', int),
    ('stacks', int),
    ('layer_size', int),
    
    ('attention_heads', int),
    ('attention_embedding_size', int),


    ('flow_type', str),
    

]


hp_dict = {   hp_name : hp_type(sys.argv[i+1]) for i, (hp_name, hp_type) in enumerate(hp_names)   }

run_id = hp_dict['run_id']
dataset_name = hp_dict['dataset_name']

dataset_name = hp_dict['dataset_name']
learning_rate = hp_dict['learning_rate']


blocks = block_types[hp_dict['blocks']] 


stacks = hp_dict['stacks']
layer_size = hp_dict['layer_size']


attention_heads = hp_dict['attention_heads']
attention_embedding_size = hp_dict['attention_embedding_size']


flow_type = hp_dict['flow_type']


# hyperparameters not tuned
max_learning_rate = 1e-3
hp_dict['max_learning_rate'] = max_learning_rate



linear_layers = 0
hp_dict['linear_layers'] = linear_layers

attention_layers = 1
hp_dict['attention_layers'] = attention_layers

stack_features_along_time = 0
hp_dict['stack_features_along_time'] = stack_features_along_time

positional_encoding = 1
hp_dict['positional_encoding'] = positional_encoding

dropout = 0.5
hp_dict['dropout'] = dropout

use_dropout_layer = 1
hp_dict['use_dropout_layer'] = use_dropout_layer

dequantize = True if dataset_name in ['solar_nips', 'taxi_30min'] else False
print('dequantize: ', dequantize)
hp_dict['dequantize'] = dequantize

modelpath = f'{run_id}'


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



# dataset, dataset_train, dataset_test = load_dataset(name=dataset_name, validation_set=False)
dataset, dataset_train, dataset_val, dataset_test, split_offset = load_dataset(name=dataset_name, train_pct=0.7)

evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                target_agg_funcs={'sum': np.sum})

target_dim=int(dataset.metadata.feat_static_cat[0].cardinality)
prediction_length = dataset.metadata.prediction_length
context_length = prediction_length * 4
freq = dataset.metadata.freq






# TimeAttentionNBeatsBlock, NBeatsBlock
estimator = NBEATSFlowEstimator(
    target_dim=target_dim,
    prediction_length=prediction_length,
    freq=freq,
    split_offset=split_offset,

    loss_function='MAPE',
    context_length=context_length,
    stack_features_along_time=stack_features_along_time,
    block=blocks,
    stacks=stacks,
    linear_layers=linear_layers,
    layer_size=layer_size,

    attention_layers=attention_layers,
    attention_embedding_size=attention_embedding_size,
    attention_heads=attention_heads,
    positional_encoding=positional_encoding,
    dropout=dropout,
    use_dropout_layer=use_dropout_layer,

    flow_type=flow_type,
    dequantize=dequantize,
    test_sampler = 'test_sampler parameter funktioniert',



    trainer=Trainer(device=device,
                    epochs=1,
                    learning_rate=learning_rate,
                    maximum_learning_rate=max_learning_rate,
                    num_batches_per_epoch=100,
                    batch_size=64,
                    clip_gradient=None,
                    weight_decay=0.01,
                    # optimizer="AdamW"
                    )
)

