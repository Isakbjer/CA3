import time
from tqdm import tqdm # Cool progress bar

import numpy as np
import pandas as pd
import cv2
import h5py

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow.keras as ks

# SEED = 458
# RNG = np.random.default_rng(SEED) # Random number generator

# F1-score metric
from tensorflow.keras import backend as K

def plot_training_history(training_history_object, list_of_metrics=['Accuracy', 'F1_score', 'IoU']):
    """
    Description: This is meant to be used in interactive notebooks
    Input:
        training_history_object:: training history object returned from 
                                  tf.keras.model.fit()
        list_of_metrics        :: Can be any combination of the following options 
                                  ('Loss', 'Precision', 'Recall' 'F1_score', 'IoU'). 
                                  Generates one subplot per metric, where training 
                                  and validation metric is plotted.
    Output:
    """
    rawDF = pd.DataFrame(training_history_object.history)
    plotDF = pd.DataFrame()

    plotDF['Accuracy']     = (rawDF['true_positives'] + rawDF['true_negatives']) / (rawDF['true_positives'] + rawDF['true_negatives'] + rawDF['false_positives'] + rawDF['false_negatives'])
    plotDF['val_Accuracy'] = (rawDF['val_true_positives'] + rawDF['val_true_negatives']) / (rawDF['val_true_positives'] + rawDF['val_true_negatives'] + rawDF['val_false_positives'] + rawDF['val_false_negatives'])

    plotDF['IoU']          = rawDF['true_positives'] / (rawDF['true_positives'] + rawDF['false_positives'] + rawDF['false_negatives'])
    plotDF['val_IoU']      = rawDF['val_true_positives'] / (rawDF['val_true_positives'] + rawDF['val_false_positives'] + rawDF['val_false_negatives'])
    
    plotDF['F1_score']     = rawDF['F1_score']
    plotDF['val_F1_score'] = rawDF['val_F1_score']

    train_keys = list_of_metrics
    valid_keys = ['val_' + key for key in list_of_metrics]
    nr_plots = len(list_of_metrics)
    fig, ax = plt.subplots(1,nr_plots,figsize=(5*nr_plots,4))
    for i in range(len(list_of_metrics)):
        ax[i].plot(np.array(plotDF[train_keys[i]]), label='Training')
        ax[i].plot(np.array(plotDF[valid_keys[i]]), label='Validation')
        ax[i].set_xlabel('Epoch')
        ax[i].set_title(list_of_metrics[i])
        ax[i].grid('on')
        ax[i].legend()
    fig.tight_layout
    plt.show()

def plot_training_history_and_return(training_history_object, list_of_metrics=['Accuracy', 'F1_score', 'IoU']):
    """
    Description: This is meant to be used in scripts run on Orion
    Input:
        training_history_object:: training history object returned from 
                                  tf.keras.model.fit()
        list_of_metrics        :: Can be any combination of the following options 
                                  ('Loss', 'Precision', 'Recall' 'F1_score', 'IoU'). 
                                  Generates one subplot per metric, where training 
                                  and validation metric is plotted.
    Output:
    """
    rawDF = pd.DataFrame(training_history_object.history)
    plotDF = pd.DataFrame()

    plotDF['Accuracy']     = (rawDF['true_positives'] + rawDF['true_negatives']) / (rawDF['true_positives'] + rawDF['true_negatives'] + rawDF['false_positives'] + rawDF['false_negatives'])
    plotDF['val_Accuracy'] = (rawDF['val_true_positives'] + rawDF['val_true_negatives']) / (rawDF['val_true_positives'] + rawDF['val_true_negatives'] + rawDF['val_false_positives'] + rawDF['val_false_negatives'])

    plotDF['IoU']          = rawDF['true_positives'] / (rawDF['true_positives'] + rawDF['false_positives'] + rawDF['false_negatives'])
    plotDF['val_IoU']      = rawDF['val_true_positives'] / (rawDF['val_true_positives'] + rawDF['val_false_positives'] + rawDF['val_false_negatives'])
    
    plotDF['F1_score']     = rawDF['F1_score']
    plotDF['val_F1_score'] = rawDF['val_F1_score']

    train_keys = list_of_metrics
    valid_keys = ['val_' + key for key in list_of_metrics]
    nr_plots = len(list_of_metrics)
    fig, ax = plt.subplots(1,nr_plots,figsize=(5*nr_plots,4))
    for i in range(len(list_of_metrics)):
        ax[i].plot(np.array(plotDF[train_keys[i]]), label='Training')
        ax[i].plot(np.array(plotDF[valid_keys[i]]), label='Validation')
        ax[i].set_xlabel('Epoch')
        ax[i].set_title(list_of_metrics[i])
        ax[i].grid('on')
        ax[i].legend()
    fig.tight_layout
    return fig