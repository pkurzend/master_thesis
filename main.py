

from .models.nbeats import NBeatsBlock, MultivariateNBeatsBlock, TimeAttentionNBeatsBlock, FeatureAttentionNBeatsBlock
from .models.estimators.nbeats_estimator import NBEATSEstimator, 
from .dataset.loader import load_dataset, generate_data, plot_data, 
import sys 
from .utils import device
from .trainer import Trainer


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


# creates folders
folders = ['models', 'gridresults', 'errs', 'logs']
for f in folders:
    try:
        os.makedirs(f)
    except OSError:
        pass


dataset_name = 'traffic_nips'

dataset, dataset_train, dataset_val, dataset_test = load_dataset(name=dataset_name)

evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})


# TimeAttentionNBeatsBlock, FeatureAttentionNBeatsBlock, MultivariateNBeatsBlock, NBeatsBlock
estimator = NBEATSEstimator(
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    prediction_length=dataset.metadata.prediction_length,
    input_dim=3856,
    freq=dataset.metadata.freq,
    loss_function='MAPE',
    context_length=120,
    stack_features_along_time=False,
    block=TimeAttentionNBeatsBlock,
    stacks=6,
    linear_layers=2,
    layer_size=512//2,
    attention_layers=1,
    attention_embedding_size=512,
    attention_heads=1,
    interpretable=True,
    # use_time_features=True,
    trainer=Trainer(device=device,
                    epochs=4,
                    learning_rate=1e-5,
                    maximum_learning_rate=1e-3,
                    num_batches_per_epoch=3,
                    batch_size=64,
                    clip_gradient=1.0
                    )
)


predictor = estimator.train(dataset_train, dataset_val)


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