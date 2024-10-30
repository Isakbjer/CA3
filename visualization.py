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

def plot_training_history(training_history_object, list_of_metrics=['Accuracy', 'F1_score', 'IoU'], start_epoch=0):
    """
    Description: This is meant to be used in interactive notebooks.
    Input:
        training_history_object:: training history object returned from 
                                  tf.keras.model.fit()
        list_of_metrics        :: Can be any combination of the following options 
                                  ('Loss', 'Precision', 'Recall', 'F1_score', 'IoU'). 
                                  Generates one subplot per metric, where training 
                                  and validation metric is plotted.
        start_epoch            :: Plot metrics starting from this epoch (default: 0).
    Output:
    """
    rawDF = pd.DataFrame(training_history_object.history)
    plotDF = pd.DataFrame()

    # Extract the suffix (e.g., '_5') from one of the metric keys
    suffix = ''
    for key in rawDF.columns:
        if key.startswith('true_positives'):
            suffix = key[len('true_positives'):]
            break

    # Calculate Accuracy (TP + TN) / (TP + TN + FP + FN) if metrics are present
    if all(metric + suffix in rawDF.columns for metric in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']):
        plotDF['Accuracy'] = (rawDF['true_positives' + suffix] + rawDF['true_negatives' + suffix]) / \
                             (rawDF['true_positives' + suffix] + rawDF['true_negatives' + suffix] + rawDF['false_positives' + suffix] + rawDF['false_negatives' + suffix])
        plotDF['val_Accuracy'] = (rawDF['val_true_positives' + suffix] + rawDF['val_true_negatives' + suffix]) / \
                                 (rawDF['val_true_positives' + suffix] + rawDF['val_true_negatives' + suffix] + rawDF['val_false_positives' + suffix] + rawDF['val_false_negatives' + suffix])

    # Calculate IoU (TP) / (TP + FP + FN)
    if all(metric + suffix in rawDF.columns for metric in ['true_positives', 'false_positives', 'false_negatives']):
        plotDF['IoU'] = rawDF['true_positives' + suffix] / \
                        (rawDF['true_positives' + suffix] + rawDF['false_positives' + suffix] + rawDF['false_negatives' + suffix])
        plotDF['val_IoU'] = rawDF['val_true_positives' + suffix] / \
                            (rawDF['val_true_positives' + suffix] + rawDF['val_false_positives' + suffix] + rawDF['val_false_negatives' + suffix])

    # Add F1 score directly (custom metric)
    if 'f1_score' in rawDF.columns:
        plotDF['F1_score'] = rawDF['f1_score']
    if 'val_f1_score' in rawDF.columns:
        plotDF['val_F1_score'] = rawDF['val_f1_score']

    # Filter data to plot only after the specified start_epoch
    plotDF = plotDF.iloc[start_epoch:]
    
    # Plotting the requested metrics
    train_keys = list_of_metrics
    valid_keys = ['val_' + key for key in list_of_metrics]
    nr_plots = len(list_of_metrics)

    fig, ax = plt.subplots(1, nr_plots, figsize=(5 * nr_plots, 4))

    for i in range(len(list_of_metrics)):
        metric = list_of_metrics[i]
        if metric in plotDF.columns and 'val_' + metric in plotDF.columns:
            ax[i].plot(np.array(plotDF[train_keys[i]]), label='Training')
            ax[i].plot(np.array(plotDF[valid_keys[i]]), label='Validation')
            ax[i].set_xlabel('Epoch')
            ax[i].set_title(metric)
            ax[i].grid(True)
            ax[i].legend()
        else:
            print(f"Metric '{metric}' not found in history. Skipping...")
    
    fig.tight_layout()
    plt.show()

