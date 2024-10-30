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
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# F1-score metric
from tensorflow.keras import backend as K

# SEED = 458
# RNG = np.random.default_rng(SEED) # Random number generator


def F1_score(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    P = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (P + K.epsilon())

    Pred_P = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (Pred_P + K.epsilon())
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def get_data(folder="DAT300_Dataset_CA_3"):
    with h5py.File("DAT300_Dataset_CA_3/tree_train.h5",'r') as f:
        print('Datasets in file:', list(f.keys()))
        X_train = np.asarray(f['X'])
        y_train = np.asarray(f['y'],dtype=np.float64) # avoid clashing dtype errors

    with h5py.File("DAT300_Dataset_CA_3/tree_test.h5",'r') as g:
        print('Datasets in file:', list(g.keys()))
        kaggle_data = np.asarray(g['X'])

    return X_train, y_train, kaggle_data

def show_image_and_mask(image_index, X_train, y_train):
  """Displays an image and its corresponding mask from the training dataset.

  Args:
    image_index: The index of the image to display.
  """
  plt.figure(figsize=(10, 5))

  plt.subplot(1, 2, 1)
  plt.imshow(X_train[image_index])
  plt.title("Image")

  plt.subplot(1, 2, 2)
  plt.imshow(y_train[image_index], cmap='gray')
  plt.title("Mask")

  plt.show()

def plot_mask_coverage_distribution(y_train):
  """Plots the distribution of mean mask coverage for all images."""
  mean_coverage = np.mean(y_train, axis=(1, 2))
  plt.figure(figsize=(8, 6))
  sns.histplot(mean_coverage, bins=20, kde=True)
  plt.title("Distribution of Mean Mask Coverage")
  plt.xlabel("Mean Mask Coverage")
  plt.ylabel("Number of Images")
  plt.show()

def rescale_images(images):
    """Rescales images to the range [0, 1]."""
    if np.max(images) > 1:
      return images / np.max(images)
    else:
      return images
    
def plot_predictions(model, X, y, num_images=5):
    """Plots the model's predictions alongside the original images and masks."""
    predictions = model.predict(X[:num_images])
    for i in range(num_images):
        plt.figure(figsize=(15, 10))

        plt.subplot(1, 3, 1)
        plt.imshow(X[i])
        plt.title("Image")

        plt.subplot(1, 3, 2)
        plt.imshow(y[i], cmap='gray')
        plt.title("Mask")

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i].squeeze(), cmap='gray')
        plt.title("Model Prediction")

        plt.show()


def create_train_generator(X, Y, batch_size=32, rotation_range=20, width_shift_range=0.1, 
                           height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, 
                           horizontal_flip=True, fill_mode='nearest', seed=42):
    """
    Creates a combined generator for images and masks with specified augmentation parameters.
    
    Parameters:
    - X: numpy array of training images
    - Y: numpy array of corresponding mask images
    - batch_size: number of samples per batch
    - rotation_range: degree range for random rotations
    - width_shift_range: fraction of total width for random horizontal shift
    - height_shift_range: fraction of total height for random vertical shift
    - shear_range: shear intensity (in radians)
    - zoom_range: range for random zoom
    - horizontal_flip: randomly flip inputs horizontally
    - fill_mode: points outside boundaries are filled according to this mode
    - seed: random seed for reproducibility
    
    Returns:
    - A generator that yields augmented images and masks in each batch
    """
    # Create image data generator
    image_datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode
    )

    # Create mask data generator
    mask_datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode
    )

    # Set up generators for images and masks with the same seed
    image_generator = image_datagen.flow(X, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(Y, batch_size=batch_size, seed=seed)
    
    # Combine generators to yield images and masks together
    def combined_generator():
        for x, y in zip(image_generator, mask_generator):
            yield x, y
    
    return combined_generator()
