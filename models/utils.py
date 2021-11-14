import torch 
import numpy as np 

import torch.nn as nn
import torch.nn.functional as F
import os 
import matplotlib.pyplot as plt
import pandas as pd 
import sys 
import torch 

import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def plot_forecasts(targets, forecasts, modelpath=None, prediction_length=24, target_dim=20):

    folders = ['plots']
    for f in folders:
        try:
            os.makedirs(f)
        except OSError:
            pass

    ts_entry = targets[0]  #<-this line is needed
    forecast_entry = forecasts[0]  #<-this line is needed

    max_rows = forecast_entry.samples.shape[-1] // 4
    n_rows = min(4, max_rows)

    fig, axes = plt.subplots(n_rows, 4, figsize=(20,15))
    fig.tight_layout()


    if forecast_entry.samples.shape[0] > 1:
        multiple_samples = True
    else:
        multiple_samples = False

    
    plt.subplots_adjust(hspace = 0.3, wspace=0.2)
    for i in range(4*n_rows):
        plt.subplot(n_rows,4,i+1)
        # fig.add_subplot(4, 4, i+1)
        ts_entry[i][-50:].plot(label='Observations')

        if not multiple_samples:
            s = pd.Series(forecast_entry.copy_dim(i).samples[0])
            s.index = ts_entry[i][-s.shape[0]:].index
            s.plot(label='Forecasts', color='g')

        else:
            forecast_entry.copy_dim(i).plot(color='g', label=' Forecasts', **{'figure' : fig})
    plt.legend(loc="best")
    if modelpath is None:
        return None
    plt.savefig(F'plots/{modelpath}_2.png')





def plot1(targets, forecasts, modelpath=None, prediction_length=24, target_dim=20, plots_folder=None):
    ts_entry = targets[0]  #<-this line is needed
    forecast_entry = forecasts[0]  #<-this line is needed


    fig, axes = plt.subplots(10, 4, figsize=(20,30))

    axes = [ax for row in axes for ax in row]



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
        l2, = axes[j].plot(model_x, np.median(np.array(forecast_entry.copy_dim(j).samples), axis=0), color='g')
    if plots_folder is None or modelpath is None:
        return
    plt.savefig(F'{plots_folder}/{modelpath}_1.png')

def plot2(targets, forecasts, modelpath=None, prediction_length=24, target_dim=20, plots_folder=None):
    ts_entry = targets[0]  #<-this line is needed
    forecast_entry = forecasts[0]  #<-this line is needed
    for i in range(4):
        plt.subplot(2,2,i+1)
        ts_entry[i][-120:].plot()
        forecast_entry.copy_dim(i).plot(color='g')
    if plots_folder is None or modelpath is None:
        return
    plt.savefig(F'{plots_folder}/{modelpath}_2.png')




def plot_learning_curves(train_losses, val_losses=None, level='epoch', ylim=None, fname=''):
    # ylim: (ymin, ymax)
    # train_losses: np array (n_epochs, batches_per_epoch)
    # val_losses: np array (n_epochs, batches_per_epoch)

    
    # creates folders
    folders = ['plots']
    for f in folders:
        try:
            os.makedirs(f)
        except OSError:
            pass
    
    
    if len(train_losses.shape) == 1:
        # only one learning curve, no standard dev
        n_epochs = train_losses.shape[0]
    elif len(train_losses.shape)==2:
        n_epochs, n_runs = train_losses.shape

    else:
        raise('Losses wrong shape')

    fig, axes = plt.subplots(1, 1, figsize=(8,5)) # nrows, ncols

    xlabel = 'Epoch' if level == 'epoch' else 'Batch'

    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel(xlabel)
    axes.set_ylabel("Loss")

    x = list(range(n_epochs))

    if len(train_losses.shape) == 1:
        axes.plot(
            x, train_losses, "-", color="orange", label="Training Loss"
        )
        if val_losses is not None:
            axes.plot(
                x, val_losses, "-", color="blue", label="Validation Loss"
            )

    elif len(train_losses.shape)==2:

        train_losses_mean = np.mean(train_losses, axis=1) 
        train_losses_std = np.std(train_losses, axis=1)
        if val_losses is not None:
            val_losses_mean = np.mean(val_losses, axis=1)
            val_losses_std = np.std(val_losses, axis=1)

        

        assert train_losses_mean.shape[0] == n_epochs


            # Plot learning curve
        axes.grid()
        axes.fill_between(
            x,
            train_losses_mean - train_losses_std,
            train_losses_mean + train_losses_std,
            alpha=0.1,
            color="o",
        )

        axes.plot(
            x, train_losses_mean, "-", color="orange", label="Training Loss"
        )

        if val_losses is not None:
            axes.fill_between(
                x,
                val_losses_mean - val_losses_std,
                val_losses_mean + val_losses_std,
                alpha=0.1,
                color="b",
            )

            axes.plot(
                x, val_losses_mean, "-", color="blue", label="Validation Loss"
            )
    axes.legend(loc="best")

    plt.savefig(f'plots/learningcurves_{fname}.pdf', format='pdf')
    plt.clf()
