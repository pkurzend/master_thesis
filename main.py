

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


print('__Number CUDA Devices:', torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.get_device_name(0))
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])


# creates folders
folders = ['models', 'gridresults', 'errs', 'logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass


dataset_name = 'traffic_nips'

dataset, dataset_train, dataset_val, dataset_test, split_offset = load_dataset(name=dataset_name, train_pct=0.7)

evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})


# TimeAttentionNBeatsBlock, FeatureAttentionNBeatsBlock, MultivariateNBeatsBlock, NBeatsBlock
estimator = NBEATSEstimator(
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    prediction_length=dataset.metadata.prediction_length,
    input_dim=3856,
    freq=dataset.metadata.freq,
    split_offset=split_offset,
    loss_function='MAPE',
    context_length=120,
    stack_features_along_time=True,
    block=TimeAttentionNBeatsBlock,
    stacks=24,
    linear_layers=2,
    layer_size=512,
    attention_layers=1,
    attention_embedding_size=512,
    attention_heads=4,
    interpretable=False,
    # use_time_features=True,
    trainer=Trainer(device=device,
                    epochs=4,
                    learning_rate=1e-5,
                    maximum_learning_rate=1e-3,
                    num_batches_per_epoch=100,
                    batch_size=32,
                    clip_gradient=1.0
                    )
)


predictor = estimator.train(dataset_train, dataset_val, num_workers=4)

train_losses = estimator.history['train_epoch_losses']
val_losses = estimator.history['val_epoch_losses']
for train_loss, val_loss in zip(train_losses, val_losses):
    print('Train loss: ', train_loss, '\t', 'Val loss: ', val_loss)



forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_val,
                                             predictor=predictor,
                                             num_samples=1)


forecasts = list(forecast_it)
targets = list(ts_it)

print(len(forecasts))
print(len(targets))
import sys 
sys.stdout.flush()

agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))



print("CRPS: {}".format(agg_metric['mean_wQuantileLoss']))
print("ND: {}".format(agg_metric['ND']))
print("NRMSE: {}".format(agg_metric['NRMSE']))
print("MSE: {}".format(agg_metric['MSE']))

print("CRPS-Sum: {}".format(agg_metric['m_sum_mean_wQuantileLoss']))
print("ND-Sum: {}".format(agg_metric['m_sum_ND']))
print("NRMSE-Sum: {}".format(agg_metric['m_sum_NRMSE']))
print("MSE-Sum: {}".format(agg_metric['m_sum_MSE']))