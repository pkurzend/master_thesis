import time
from typing import List, Optional, Union, NamedTuple
import pts
from tqdm import tqdm

import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from gluonts.core.component import validated
from pts import Trainer
import uuid
# print uuid.uuid4()

from ..nbeats import generate_model, NBEATSTrainingNetwork, NBEATSPredictionNetwork, NBeatsBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO:
# if loss goes up or to nan, reduce learning rate and try again



# implement model parallelism like in 
# https://glassboxmedicine.com/2020/03/04/multi-gpu-training-in-pytorch-data-and-model-parallelism/
# https://www.run.ai/guides/multi-gpu/pytorch-multi-gpu-4-techniques-explained/
        
class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def unParallelize(self):
        return self.module 



class Trainer(Trainer):
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        maximum_learning_rate: float = 1e-5,
        restore_best: bool = True,
        clip_gradient: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> None:

        print('new Trainer')
       
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.clip_gradient = clip_gradient
        self.device = device
        self.restore_best = restore_best


    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:
        

        print('calling new trainer################################')

        self.first_train_batch = True
        self.first_val_batch = True
        
        optimizer = Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )


        # make model use several gpus
        if torch.cuda.device_count() > 1:
            net = MyDataParallel(net)

        best_loss = np.inf

        if os.path.exists('models/'):
            best_model_file_name = 'models/' + str(uuid.uuid4())+'best-model.pt'
        else:
            best_model_file_name = str(uuid.uuid4())+'best-model.pt'


        nan_loss = False

        # store losses
        train_losses = []
        train_epoch_losses = []
        val_losses = [] 
        val_epoch_losses = [] 

        for epoch_no in range(self.epochs):
            # mark epoch start time
            tic = time.time()
            avg_epoch_loss = 0.0


            if validation_iter is not None:
                avg_epoch_loss_val = 0.0
                val_batch_no = 1


            with tqdm(train_iter) as it:
                # print('it', next(iter(train_iter)))    

                for batch_no, data_entry in enumerate(it, start=1):

                    optimizer.zero_grad()


                    # if self.first_train_batch:
                    #     self.first_train_batch = False
                    #     for name, t in data_entry.items():
                    #       print('train ', name, t.shape)                    

                    # validation_loss
                    if validation_iter is not None:
                        with torch.no_grad():
                            val_data_entry = next(iter(validation_iter))
                            # print(val_data_entry)
                            # if self.first_val_batch:
                            #     self.first_val_batch = False
                            #     for name, t in val_data_entry.items():
                            #       print('val ', name, t.shape)

                            inputs_val = [v.to(self.device) for v in val_data_entry.values()]


                            output_val = net(*inputs_val)

                            if isinstance(output_val, (list, tuple)):
                                loss_val = output_val[0]
                            else:
                                loss_val = output_val
                            # print('loss val ', loss_val, loss_val.shape)
                            if len(loss_val.shape) > 0: #false for torch.tensor(1.4).shape = torch.size([]), true for torch.tensor([1.4, 1.5]).shape = torch.size([2,])
                                loss_val = loss_val.mean()
                            # print('loss val mean ', loss_val)
                            avg_epoch_loss_val += loss_val.item()


                    inputs = [v.to(self.device) for v in data_entry.values()]
                    
                    # if not np.isfinite(ndarray.sum(loss).asscalar()):
  
                    output = net(*inputs)
                    if isinstance(output, (list, tuple)):
                        loss = output[0]
                    else:
                        loss = output

                    if len(loss.shape) > 0:
                        loss = loss.mean()
                    avg_epoch_loss += loss.item()

                    nan_halt_cond =  (torch.isnan(loss).any()) or (torch.isnan(loss_val).any()) if validation_iter is not None else torch.isnan(loss).any()
                    if nan_halt_cond:
                        nan_loss = True
                        print('NAN LOSS OCCURED, ABORT TRAINING')
                        break

                    if validation_iter is not None:
                        post_fix_dict = {
                                "avg_epoch_loss": avg_epoch_loss / batch_no,
                                "avg_epoch_loss_val": avg_epoch_loss_val / batch_no,
                                "epoch": epoch_no,
                        }
                      

                        val_losses.append(avg_epoch_loss_val / batch_no)

                    else:
                        post_fix_dict={
                                "avg_epoch_loss": avg_epoch_loss / batch_no,
                                "epoch": epoch_no,
                        }
                    
                    it.set_postfix(ordered_dict=post_fix_dict, refresh=False)

                    train_losses.append(avg_epoch_loss / batch_no)





                    loss.backward()
                    if self.clip_gradient is not None:
                        nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)

                    optimizer.step()
                    lr_scheduler.step()

                    if self.num_batches_per_epoch == batch_no:
                        break

                # save best model:
                if validation_iter is not None:
                    if avg_epoch_loss_val / self.num_batches_per_epoch < best_loss:
                        torch.save(net.state_dict(), best_model_file_name)
                        best_loss = avg_epoch_loss_val / self.num_batches_per_epoch
                else:
                    if avg_epoch_loss / self.num_batches_per_epoch < best_loss:
                        torch.save(net.state_dict(), best_model_file_name)
                        best_loss = avg_epoch_loss / self.num_batches_per_epoch

            # mark epoch end time and log time cost of current epoch
            toc = time.time()


            # save epoch losses
            train_epoch_losses.append(avg_epoch_loss / self.num_batches_per_epoch)
            if validation_iter is not None:
                val_epoch_losses.append(avg_epoch_loss_val / self.num_batches_per_epoch)

            if nan_loss: 
                break




        # restore best model
        net.load_state_dict(torch.load(best_model_file_name))
        
        os.remove(best_model_file_name)

        # this must come after restoring best model otherwise parameter names in stat_dict do not match
        if torch.cuda.device_count() > 1:
            net = net.unParallelize()
        

        return train_losses, train_epoch_losses, val_losses, val_epoch_losses



import time
from typing import List, Optional, Union

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from gluonts.core.component import validated


class Trainer_pts:
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        maximum_learning_rate: float = 1e-2,
        clip_gradient: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.clip_gradient = clip_gradient
        self.device = device

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:
        optimizer = Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )

        for epoch_no in range(self.epochs):
            # mark epoch start time
            tic = time.time()
            cumm_epoch_loss = 0.0
            total = self.num_batches_per_epoch - 1

            # training loop
            with tqdm(train_iter, total=total) as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()

                    inputs = [v.to(self.device) for v in data_entry.values()]
                    output = net(*inputs)

                    if isinstance(output, (list, tuple)):
                        loss = output[0]
                    else:
                        loss = output

                    cumm_epoch_loss += loss.item()
                    avg_epoch_loss = cumm_epoch_loss / batch_no
                    it.set_postfix(
                        {
                            "epoch": f"{epoch_no + 1}/{self.epochs}",
                            "avg_loss": avg_epoch_loss,
                        },
                        refresh=False,
                    )

                    loss.backward()
                    if self.clip_gradient is not None:
                        nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)

                    optimizer.step()
                    lr_scheduler.step()

                    if self.num_batches_per_epoch == batch_no:
                        break
                it.close()

            # validation loop
            if validation_iter is not None:
                cumm_epoch_loss_val = 0.0
                with tqdm(validation_iter, total=total, colour="green") as it:

                    for batch_no, data_entry in enumerate(it, start=1):
                        inputs = [v.to(self.device) for v in data_entry.values()]
                        with torch.no_grad():
                            output = net(*inputs)
                        if isinstance(output, (list, tuple)):
                            loss = output[0]
                        else:
                            loss = output

                        cumm_epoch_loss_val += loss.item()
                        avg_epoch_loss_val = cumm_epoch_loss_val / batch_no
                        it.set_postfix(
                            {
                                "epoch": f"{epoch_no + 1}/{self.epochs}",
                                "avg_loss": avg_epoch_loss,
                                "avg_val_loss": avg_epoch_loss_val,
                            },
                            refresh=False,
                        )

                        if self.num_batches_per_epoch == batch_no:
                            break

                it.close()

            # mark epoch end time and log time cost of current epoch
            toc = time.time()