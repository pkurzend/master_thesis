from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
# from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
from gluonts.dataset.split import OffsetSplitter
from gluonts.dataset.common import ListDataset

import numpy as np
import matplotlib.pyplot as plt
import torch 

from gluonts.dataset.split import OffsetSplitter
from gluonts.dataset.common import ListDataset



def load_dataset(name='traffic_nips', validation_set=True, train_pct=0.7):

    dataset = get_dataset("traffic_nips", regenerate=False)
    print(dataset.train.__dict__)
    print(dataset.test.__dict__)

    train_grouper = MultivariateGrouper(max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))
    test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                   max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality))

    if validation_set:
        split_offset=int(next(iter(dataset.train))['target'].shape[0]*train_pct)
        prediction_length=next(iter(dataset.train))['target'].shape[0]
        splitter = OffsetSplitter(split_offset=split_offset, prediction_length=prediction_length-split_offset)
        dataset_train, dataset_val = splitter.split(dataset.train)

        dataset_train = ListDataset(dataset_train[1], freq=dataset.metadata.freq, one_dim_target=False) # listDataset
        dataset_val = ListDataset(dataset_val[1], freq=dataset.metadata.freq, one_dim_target=False) # listDataset

    else:   
        dataset_train = train_grouper(dataset.train)

    dataset_test = test_grouper(dataset.test)  # ListDataset  but with target 2 dimensional
    print(len(dataset_train))
    print(len(dataset_train.list_data))
    print(dataset_train.list_data[0])
    print(len(dataset_train.list_data[1]))

    print(len(dataset_val))
    print(len(dataset_val.list_data))
    print(dataset_val.list_data[0])
    print(len(dataset_val.list_data[1]))

    dataset_train.list_data = [
    {'feat_static_cat': [0],
     'start': dataset_train.list_data[0]['start'],
     'target' : np.stack([ts['target'] for ts in dataset_train.list_data])
     }                 
    ]

    dataset_val.list_data = [
        {'feat_static_cat': [0],
        'start': dataset_val.list_data[0]['start'],
        'target' : np.stack([ts['target'] for ts in dataset_val.list_data])
        }                 
    ]


    print(dataset_train.list_data[0]['target'].shape)
    print(dataset_val.list_data[0]['target'].shape)
    print(dataset_test.list_data[0]['target'].shape)

    if validation_set:
        return dataset, dataset_train, dataset_val, dataset_test, split_offset
    else: 
        return dataset, dataset_train, dataset_test


def plot_data(dataset, max_len=500):
    print(dataset.__dict__)
    x = dataset.list_data[0]['target']
    print(x.shape)
    print((x==0).all())
    
    plt.figure(figsize=(30, 10))


    for ts in x:
        plt.plot(ts[:max_len])
    plt.show()


# own data loader function:
def generate_data(data, context_length=3*24, prediction_length=24, every_nth=None, bs=64, normalize=True, endless_mode=False):
    """
    @data: numpy array or torch tensor of shape (E, S) where S is the whole sequence length
    @context_length: time series length, trainling datapoints
    @prediction_length: predict next y_len elements
    @every_nth: offsetfor rolling down the dataframe
    @bs: batchsize
    @normalize: apply mean normalization (divide src and trg by src mean)
    @yields: tuple of x and y data: pytorch tensors of shape (S, N, E) and (T, N, E) (x_len, bs, n_features) and  (y_len , bs, n_features)

    pytorch transformer requires input shapes of source sequence and target sequence: 
    src: (S,N,E)
    tgt: (T,N,E)
    T = target sequence length
    S = source sequence length
    N = batch size
    E = n_features (ie embedding size)

    with encoder input (x 1 , x 2 , ..., x 10 ) and the decoder input (x 10 , ..., x 13 ),
    the decoder aims to output (x 11 , ..., x 14 )
    """
    x_len = context_length
    y_len = prediction_length
    if every_nth is None:
      every_nth = y_len

    batch_src = []
    batch_trg = []

    # print('data shape: ', data.shape)
    

    for i in range(0, data.shape[0] - x_len-y_len - 1, every_nth): 
        window = data[i:i+x_len+y_len]
        assert window.shape[0] == x_len+y_len
        assert x_len+y_len+i < data.shape[0]
        ts = window[:x_len] # keeps extra dimension for features (nx1)
        targets = window[x_len:]
        assert ts.shape[0] == x_len
        assert targets.shape[0] == y_len
        batch_src.append(torch.tensor(ts, dtype=torch.float))
        batch_trg.append(torch.tensor(targets, dtype=torch.float))

        if len(batch_src) == bs:
            x, y = torch.stack(batch_src, dim=1), torch.stack(batch_trg, dim=1) # (S, N, E) and (T, N, E)
            if normalize:
                mean = x.mean(dim=0)
                x = x/mean
                y = y/mean
            yield (x, y)
            batch_src = []
            batch_trg = []

'''
example usge:
dataset, dataset_train, dataset_val, dataset_test = load_dataset()
data = generate_data(dataset_train.list_data[0]['target'].T, 
                     context_length=dataset.metadata.prediction_length*3, 
                     prediction_length=dataset.metadata.prediction_length,
                     every_nth = 1
                    )

data = list(data)

'''