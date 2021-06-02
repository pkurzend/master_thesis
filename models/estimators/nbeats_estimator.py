from .estimator_base import PyTorchEstimator 
from typing import List, Optional

import torch
import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature
from gluonts.torch.support.util import copy_parameters
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.predictor import Predictor
from gluonts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    InstanceSampler,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    RemoveFields,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)

# from pts import Trainer
# from pts.model import PyTorchEstimator
from pts.model.utils import get_module_forward_input_names
from pts.feature import (
    fourier_time_features_from_frequency,
    lags_for_fourier_time_features_from_frequency,
)

from .estimator_base import PyTorchEstimator 
from ..nbeats import generate_model, NBEATSTrainingNetwork, NBEATSPredictionNetwork, NBeatsBlock


from ...utils import device



class PredictionSplitSampler(InstanceSampler):
    """
    Sampler used for prediction. Always selects the last time point for
    splitting i.e. the forecast point for the time series.
    """

    allow_empty_interval: bool = False

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts) # returns         return (self.min_past,ts.shape[self.axis] - self.min_future,)
        # print('PredictionSplitSampler ab ', a, b)
            
            
        

        # print('PredictionSplitSampler ', ts.shape) # contains the whole time series matrix: all time steps and all time series (featresu): shape: (963, 4001)
        assert self.allow_empty_interval or a <= b
        result = np.array([b]) if a <= b else np.array([], dtype=int)
        # print('PredictionSplitSampler ', result)
        return result


def ValidationSplitSampler(
    axis: int = -1, min_past: int = 0, min_future: int = 0
) -> PredictionSplitSampler:
    return PredictionSplitSampler(
        allow_empty_interval=True,
        axis=axis,
        min_past=min_past,
        min_future=min_future,
    )



class ExpectedNumInstanceSampler(ExpectedNumInstanceSampler):
    """
    Keeps track of the average time series length and adjusts the probability
    per time point such that on average `num_instances` training examples are
    generated per time series.
    Parameters
    ----------
    num_instances
        number of training examples generated per time series on average
    """

    num_instances: float
    total_length: int = 0
    n: int = 0

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts) # returns   (self.min_past,ts.shape[self.axis] - self.min_future,) = (history_length, prediction_length)
        # print('ExpectedNumInstanceSampler ab ', a, b)
        window_size = b - a + 1

        # print('window_size ', window_size) # 3690

        # print('ExpectedNumInstanceSampler ', ts.shape) # contains the whole time series matrix: all time steps and all time series (featresu): shape: (963, 2800)

        if window_size <= 0:
            return np.array([], dtype=int)

        self.n += 1
        self.total_length += window_size
        avg_length = self.total_length / self.n

        if avg_length <= 0:
            return np.array([], dtype=int)

        p = self.num_instances / avg_length
        # print('ExpectedNumInstanceSampler p ', p)
        (indices,) = np.where(np.random.random_sample(window_size) < p) # random_sample returns random numbers between 0 and 1
        # print('ExpectedNumInstanceSampler indices ', indices.shape, indices+a)
        return indices + a

class NBEATSEstimator(PyTorchEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        target_dim: int,
        trainer=Trainer(),#: Trainer = Trainer(),
        context_length: Optional[int] = None,
        input_dim: Optional[int] = None,
        loss_function: Optional[str] = "MAPE",
        stacks: int=30, 
        linear_layers: int=4, 
        layer_size: int=512, 
        block : nn.Module = NBeatsBlock,  
        attention_layers : int=1, 
        attention_embedding_size : int=512, 
        attention_heads : int = 1,

        # parameters for interpretable verions
        interpretable : bool = False,
        degree_of_polynomial : int = 3,
        trend_layer_size : int = 256,
        seasonality_layer_size : int = 2048,
        num_of_harmonics : int = 1,

        use_feat_dynamic_real: bool = False,
        stack_features_along_time : bool = False, # if False, data has shape (bacht_size, context_length, input_dim) where input_dim = target_dim + n_lags*target_dim + embedding_dim*target_dim + n_time_features
        compute_input_dim : bool = True,


        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        **kwargs,
    ) -> None:
        super().__init__(trainer=trainer, **kwargs)


        # self.input_dim = input_dim if input_dim is not None else target_dim

        self.loss_function = loss_function
        self.block = block
        self.stacks=stacks
        self.linear_layers=linear_layers
        self.layer_size=layer_size
        self.attention_layers=attention_layers
        self.attention_embedding_size=attention_embedding_size
        self.attention_heads=attention_heads

        self.interpretable=interpretable
        self.degree_of_polynomial=degree_of_polynomial
        self.trend_layer_size=trend_layer_size
        self.seasonality_layer_size=seasonality_layer_size 
        self.num_of_harmonics=num_of_harmonics 




        self.freq = freq


        
        self.use_feat_dynamic_real = use_feat_dynamic_real



        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else lags_for_fourier_time_features_from_frequency(freq_str=freq)
        )

        self.time_features = (
            time_features
            if time_features is not None
            else fourier_time_features_from_frequency(self.freq)
        )

        self.prediction_length = prediction_length
        self.target_dim = target_dim


        self.compute_input_dim = compute_input_dim
        self.stack_features_along_time = stack_features_along_time

        if not self.stack_features_along_time: # stack time features along embedding_dimension in (N, S, input_dim)
          if self.compute_input_dim or input_dim is None:
              self.input_dim = len(self.lags_seq)*self.target_dim + 2*len(self.time_features) + self.target_dim * 1 # last number is embedding_size used in nbeatsnetwork
          else:
            self.input_dim = input_dim
            

        else: # stack time features along time axis (N, input_length, target_dim)
            self.input_dim = target_dim
            

        self.context_length = (
            context_length if context_length is not None else prediction_length
        )

        


        self.history_length = self.context_length + max(self.lags_seq)
        self.pick_incomplete = pick_incomplete

        # https://github.com/awslabs/gluon-ts/blob/76fb746121e8b67c4b6b59db01f8ad682a3005e5/src/gluonts/transform/sampler.py#L23
        # An InstanceSampler is called with the time series ``ts``, 
        # Returns a set of indices at which training instances will be generated.
        # The sampled indices ``i`` satisfy ``a <= i <= b``, where ``a = min_past``
        # and ``b = ts.shape[axis] - min_future`` min_future.
        # min_past=history_length = 288
        # min_future=prediction_length=24
        

        # A splitSampler returns the index at which a time series window is splitted into context window and forecast window
        # source code above
        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0, # number of training examples generated per time series on average
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

        # returns the index of b
        # A splitSampler returns the index at which a time series window is splitted into context window and forecast window
        # self.validation_sampler = ValidationSplitSampler(
        #     min_past=0 if pick_incomplete else self.history_length,
        #     min_future=prediction_length,
        # )

        self.validation_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0, # number of training examples generated per time series on average
            min_past=split_offset,
            min_future=prediction_length,
        )

        print('history_length, context_length ', self.history_length, self.context_length)
        print('lags ', self.lags_seq)
        print('time features ', self.time_features)

    def create_transformation(self) -> Transformation:
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        
        return Chain(
            [
                RemoveFields(field_names=remove_field_names),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
                ),
                # maps the target to (1, T)
                # if the target data is uni dimensional
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                            ),
                # Replaces missing values in a numpy array (NaNs) with a dummy value (0) and adds
                # an "observed"-indicator that is ``1`` when values are observed and ``0``
                # when values are missing.
                # adds following fields:
                # past_observed_values (batch_size, history_length, target_dim)    
                # future_observed_values (batch_size, prediction_length, target_dim)
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
             
                # Adds a set of time features., adds past_time_feat and future_time_feat
                # past_time_feat: (batch_size, history_length, num_features)
                # future_time_feat: (batch_size, prediction_length, num_features)
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.use_feat_dynamic_real
                        else []
                    ),
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
             
                
                # Indices of the target dimension (batch_size, target_dim)
                # just counts from 1 to target_dim for each instance in batch
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        # samples instances returned from instance_sampler who returns the index of the point where time series is splittet into context and prediction window
        # adds a field 'past_is_pad' that indicates whether values where padded or not.
        # past_is_pad is 1 where timeseries is padded and 0 else
        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            # determines the length of the time series that is input to forward of NBEATSNetwork, 
            # this is history_length to compute lag features, later the time series is truncated at context_length
            past_length=self.history_length, 
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        )



     
    def create_training_network(self, device: torch.device) -> NBEATSTrainingNetwork:
        return NBEATSTrainingNetwork(
            loss_function=self.loss_function, 
            input_dim=self.input_dim, 
            target_dim=self.target_dim,
            context_length=self.context_length, 
            prediction_length=self.prediction_length,
            stack_features_along_time=self.stack_features_along_time, 
            freq=self.freq,
            history_length=self.history_length,
            lags_seq=self.lags_seq,
            block=self.block, 
            stacks=self.stacks,
            linear_layers=self.linear_layers,
            layer_size=self.layer_size,
            attention_layers=self.attention_layers,
            attention_embedding_size=self.attention_embedding_size,
            attention_heads=self.attention_heads,
            interpretable=self.interpretable,
            degree_of_polynomial=self.degree_of_polynomial,
            trend_layer_size=self.trend_layer_size,
            seasonality_layer_size=self.seasonality_layer_size,
            num_of_harmonics=self.num_of_harmonics
        ).to(device)
            



    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: nn.Module,
        device: torch.device,
    ) -> Predictor:
        prediction_network = NBEATSPredictionNetwork(
            loss_function=self.loss_function, 
            input_dim=self.input_dim, 
            target_dim=self.target_dim,
            context_length=self.context_length, 
            prediction_length=self.prediction_length,
            stack_features_along_time=self.stack_features_along_time,  
            freq=self.freq,
            history_length=self.history_length,
            lags_seq=self.lags_seq,
            block=self.block, 
            stacks=self.stacks,
            linear_layers=self.linear_layers,
            layer_size=self.layer_size,
            attention_layers=self.attention_layers,
            attention_embedding_size=self.attention_embedding_size,
            attention_heads=self.attention_heads,
            interpretable=self.interpretable,
            degree_of_polynomial=self.degree_of_polynomial,
            trend_layer_size=self.trend_layer_size,
            seasonality_layer_size=self.seasonality_layer_size,
            num_of_harmonics=self.num_of_harmonics
        ).to(device)

        copy_parameters(trained_network, prediction_network)
        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )

