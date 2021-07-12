
import sys
import os
sys.path.append('../../')



import numpy as np
import pandas as pd

import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator

from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

import matplotlib.pyplot as plt

print('__Number CUDA Devices:', torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

plots_folder = 'plots3'

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
    ('input_size', int),
    ('model_type', str),
    ('own_trainer', int)

]

print(hp_names)
print(sys.argv)

hp_dict = {   hp_name : hp_type(sys.argv[i+1]) for i, (hp_name, hp_type) in enumerate(hp_names)   }

dataset_name = hp_dict['dataset_name']
input_size = hp_dict['input_size']
model_type = hp_dict['model_type']
own_trainer = hp_dict['own_trainer']


if own_trainer == 1:
    from models.estimators.trainer import Trainer
else:
    from pts import Trainer


modelpath = '&'.join([F"{param}={value}" for param, value in hp_dict.items() if param not in ['stack_features_along_time', 'attention_embedding_size', 'attention_heads', 'attention_layers']])


print('hyperparameter dict: ')
print(hp_dict)

print('DATASET NAME: ', dataset_name)
print('INPUT SIZE: ', input_size)

start = '''
############################################################
###################### START TRAINING ######################
############################################################
# '''
print(start)


dataset = get_dataset(dataset_name, regenerate=False)

train_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                   max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)


evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})


print('TARGET DIM: ', int(dataset.metadata.feat_static_cat[0].cardinality))

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




result_obj = {
    'metrics' : {},
    'hyperparameters' : hp_dict,
    'dataset_name' : dataset_name,
    'model_type' : model_type,
}



if model_type == 'GRU-Real-NVP':
    # GRU-Real-NVP
    print('GRU-Real-NVP')
    estimator = TempFlowEstimator(
        target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
        prediction_length=dataset.metadata.prediction_length,
        cell_type='GRU',
        input_size=input_size,
        freq=dataset.metadata.freq,
        scaling=True,
        dequantize=True,
        n_blocks=4,
        trainer=Trainer(device=device,
                        epochs=25,
                        learning_rate=1e-3,
                        maximum_learning_rate=1e-2,
                        num_batches_per_epoch=100,
                        batch_size=64)
    )

elif model_type == 'GRU-MAF':

    # GRU-MAF
    print('GRU-MAF')
    estimator = TempFlowEstimator(
        target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
        prediction_length=dataset.metadata.prediction_length,
        cell_type='GRU',
        input_size=input_size,
        freq=dataset.metadata.freq,
        scaling=True,
        dequantize=True,
        flow_type='MAF',
        trainer=Trainer(device=device,
                        epochs=25,
                        learning_rate=1e-3,
                        maximum_learning_rate=1e-2,
                        num_batches_per_epoch=100,
                        batch_size=64)
    )

elif model_type == 'Transformer-MAF':
    # Transformer-MAF
    print('Transformer-MAF')
    estimator = TransformerTempFlowEstimator(
        d_model=16,
        num_heads=4,
        input_size=input_size,
        target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
        prediction_length=dataset.metadata.prediction_length,
        context_length=dataset.metadata.prediction_length*4,
        flow_type='MAF',
        dequantize=True,
        freq=dataset.metadata.freq,
        trainer=Trainer(
            device=device,
            epochs=14,
            learning_rate=1e-3,
            maximum_learning_rate=1e-2,
            num_batches_per_epoch=100,
            batch_size=64,
        )
    )

predictor = estimator.train(dataset_train)
forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                             predictor=predictor,
                                             num_samples=100)
forecasts = list(forecast_it)
targets = list(ts_it)

agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

plot1(targets, forecasts, modelpath)
plot2(targets, forecasts, modelpath)

print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
print("ND: {}".format(agg_metric['ND']))
print("NRMSE: {}".format(agg_metric['NRMSE']))
print("MSE: {}".format(agg_metric['MSE']))


print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))


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


result_obj['metrics'] = {
        'mse' : mse,
        'mase' : mase,
        'mape' : mape,
        'smape' : smape,
        'm_sum_mse' : mse_sum,
        'm_sum_mase' : mase_sum,
        'm_sum_mape' : mape_sum,
        'm_sum_smape' : smape_sum
    }

import pickle 
filename = F"gridresults/{modelpath}.pkl"
with open(filename, 'wb') as f:
  pickle.dump(result_obj, f)





