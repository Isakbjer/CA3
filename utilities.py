import time
from tqdm import tqdm # Cool progress bar

import numpy as np
import pandas as pd
import cv2
import h5py

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow.keras as ks
from scipy.optimize import minimize_scalar

# SEED = 458
# RNG = np.random.default_rng(SEED) # Random number generator

# F1-score metric
from tensorflow.keras import backend as K

def F1_score(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    P = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (P + K.epsilon())

    Pred_P = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (Pred_P + K.epsilon())
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
