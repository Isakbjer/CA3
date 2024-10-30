import time
from tqdm import tqdm # Cool progress bar
import random
import numpy as np
import pandas as pd
import h5py
import cv2
import re
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import *
from visualization import *
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

SEED = 458 # Feel free to set another seed if you want to
RNG = np.random.default_rng(SEED) # Random number generator
tf.random.set_seed(SEED)

with h5py.File("DAT300_Dataset_CA_3/tree_train.h5",'r') as f:
    print('Datasets in file:', list(f.keys()))
    X_train = np.asarray(f['X'])
    y_train = np.asarray(f['y'],dtype=np.float64) # avoid clashing dtype errors
    print('Nr. train images: %i'%(X_train.shape[0]))

print(X_train.shape)
print(y_train.shape)

with h5py.File("DAT300_Dataset_CA_3/tree_test.h5",'r') as g:
    print('Datasets in file:', list(g.keys()))
    kaggle_data = np.asarray(g['X'])

def rescale_images(images):
    """Rescales images to the range [0, 1]."""
    if np.max(images) > 1:
      return images / np.max(images)
    else:
      return images

X_train = rescale_images(X_train)
kaggle_data = rescale_images(kaggle_data)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)

# prompt: Create an image generator to modify the input image and mask in the same way

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an image data generator
image_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create a mask data generator (similar to image generator, but no rescaling or featurewise_center)
mask_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Combine generators
def combined_generator(image_generator, mask_generator, X, Y, batch_size):
  genX = image_generator.flow(X, batch_size=batch_size, seed=SEED)
  genY = mask_generator.flow(Y, batch_size=batch_size, seed=SEED)
  for x, y in zip(genX, genY):
    yield x, y

train_generator = combined_generator(image_datagen, mask_datagen, X_train, y_train, batch_size=32)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D, Dropout, ReLU, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives

def build_efficientnetv2_unet(input_shape, num_conv_filters=64, filter_growth_rate=2, dropout_rate=0.3):
    # Input layer
    inputs = Input(shape=input_shape)

    # Load the EfficientNetV2 model without the fully connected layers and with pretrained ImageNet weights
    effnet_encoder = EfficientNetV2B0(input_tensor=inputs, include_top=False, weights='imagenet')

    # Extract feature maps from specific layers of EfficientNetV2 for skip connections
    skip1 = effnet_encoder.get_layer('block2a_expand_activation').output  # Shape: (64, 64, 32)
    skip2 = effnet_encoder.get_layer('block3a_expand_activation').output  # Shape: (32, 32, 48)
    skip3 = effnet_encoder.get_layer('block4a_expand_activation').output  # Shape: (16, 16, 112)
    skip4 = effnet_encoder.get_layer('block6a_expand_activation').output  # Shape: (8, 8, 192)

    # Bottleneck (deepest layer from EfficientNetV2)
    bottleneck = effnet_encoder.get_layer('top_activation').output  # Shape: (4, 4, 1280)

    # Bottleneck Conv layers
    c3 = Conv2D(num_conv_filters * filter_growth_rate * 2, (3, 3), padding='same')(bottleneck)
    c3 = ReLU()(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(num_conv_filters * filter_growth_rate * 2, (3, 3), padding='same')(c3)
    c3 = ReLU()(c3)
    c3 = BatchNormalization()(c3)

    # Decoder Block 1: Upsample from (4,4) -> (8,8), skip connection from skip4
    u4 = UpSampling2D((2, 2))(c3)
    u4 = Concatenate()([u4, skip4])  # Add skip connection
    u4 = Conv2D(num_conv_filters * filter_growth_rate * 2, (3, 3), padding='same')(u4)
    u4 = ReLU()(u4)
    u4 = BatchNormalization()(u4)

    # Decoder Block 2: Upsample from (8,8) -> (16,16), skip connection from skip3
    u5 = UpSampling2D((2, 2))(u4)
    u5 = Concatenate()([u5, skip3])  # Add skip connection to (16,16,112)
    u5 = Conv2D(num_conv_filters * filter_growth_rate * 2, (3, 3), padding='same')(u5)
    u5 = ReLU()(u5)
    u5 = BatchNormalization()(u5)

    # Decoder Block 3: Upsample from (16,16) -> (32,32), skip connection from skip2
    u6 = Concatenate()([u5, skip2])  # Add skip connection
    u6 = UpSampling2D((2, 2))(u6)
    # skip2_upsampled = UpSampling2D((2, 2))(skip2)  # Additional upsampling to match shape
    u6 = Conv2D(num_conv_filters * filter_growth_rate, (3, 3), padding='same')(u6)
    u6 = ReLU()(u6)
    u6 = BatchNormalization()(u6)

    # Decoder Block 4: Upsample from (32,32) -> (64,64), skip connection from skip1
    u7 = Concatenate()([u6, skip1])  # Add skip connection
    u7 = UpSampling2D((2, 2))(u6)
    # skip1_upsampled = UpSampling2D((2, 2))(skip1)  # Additional upsampling to match shape
    u7 = Conv2D(num_conv_filters, (3, 3), padding='same')(u7)
    u7 = ReLU()(u7)
    u7 = BatchNormalization()(u7)

    # Decoder Block 5: Upsample from (64,64) -> (128,128)
    u8 = UpSampling2D((2, 2))(u7)
    u8 = Conv2D(num_conv_filters, (3, 3), padding='same')(u8)
    u8 = ReLU()(u8)
    u8 = BatchNormalization()(u8)

    # Apply Dropout if needed
    if dropout_rate > 0:
        u8 = Dropout(dropout_rate)(u8)

    # Final output layer (1-channel output for binary segmentation)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u8)

    # Build the final model
    model = Model(inputs, outputs)

    return model

# Build the model
input_shape = (128, 128, 3)
model = build_efficientnetv2_unet(input_shape, num_conv_filters=16, filter_growth_rate=2, dropout_rate=0.5)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', F1_score, TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()])

history = model.fit(train_generator,
                    validation_data=(X_val, y_val),
                    steps_per_epoch=100,
                    batch_size=32,
                    epochs=50)