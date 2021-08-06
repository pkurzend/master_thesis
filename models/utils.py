import torch 
import numpy as np 

import torch.nn as nn
import torch.nn.functional as F
import os 
import matplotlib.pyplot as plt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



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