import os
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives


# User-defined paths and parameters
USER_NAME = 'dat300-h24-3'
PATH_TO_DATASET = f'DAT300_Dataset_CA_3/tree_train.h5'
# PATH_TO_DATASET = f'/mnt/users/{USER_NAME}/tree_train.h5'
MODEL_NAME = 'efficientnet_unet_model'
BATCH_SIZE = 32
EPOCHS = 20
# PATH_TO_STORE_MODEL = f'/mnt/users/{USER_NAME}/models'
PATH_TO_STORE_MODEL = 'models'
SEED = 458
RNG = np.random.default_rng(SEED)
tf.random.set_seed(SEED)

# Define save paths
path_to_store_training_history = os.path.join(PATH_TO_STORE_MODEL, MODEL_NAME + '_training_history.png')
path_to_store_model = os.path.join(PATH_TO_STORE_MODEL, MODEL_NAME + '.keras')

# Plotting function for training history
def plot_training_history(history, metrics=None):
    history_dict = history.history
    if metrics is None:
        metrics = [key for key in history_dict.keys() if 'val_' not in key]
    df = pd.DataFrame(history_dict)
    fig, ax = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for i, metric in enumerate(metrics):
        ax[i].plot(df[metric], label='Training')
        ax[i].plot(df[f'val_{metric}'], label='Validation')
        ax[i].set_xlabel('Epoch')
        ax[i].set_title(metric)
        ax[i].grid(True)
        ax[i].legend()
    fig.tight_layout()
    return fig

# Load and preprocess data
with h5py.File(PATH_TO_DATASET, 'r') as f:
    print('Datasets in file:', list(f.keys()))
    X_train = np.asarray(f['X'], dtype=np.float32) / 255.0
    y_train = np.asarray(f['y'], dtype=np.float32)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)

# F1-score metric

def F1_score(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    P = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (P + K.epsilon())

    Pred_P = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (Pred_P + K.epsilon())
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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

# # Create a mask data generator (similar to image generator, but no rescaling or featurewise_center)
# mask_datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# Combine generators
def combined_generator(image_generator, mask_generator, X, Y, batch_size):
  genX = image_generator.flow(X, batch_size=batch_size, seed=SEED)
  genY = image_generator.flow(Y, batch_size=batch_size, seed=SEED)
  for x, y in zip(genX, genY):
    yield x, y

train_generator = combined_generator(image_datagen, image_datagen, X_train, y_train, batch_size=32)

# Build the model

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

# Train the model

start_time = time.time()

history = model.fit(train_generator,
                    validation_data=(X_val, y_val),
                    steps_per_epoch=100,
                    batch_size=32,
                    epochs=50)

training_time = time.time() - start_time

# Plot and save training history
history_plot = plot_training_history(history)
history_plot.savefig(path_to_store_training_history)

# Save model
model.save(path_to_store_model)
print(f'Model saved at {path_to_store_model}')
print(f'Training history saved at {path_to_store_training_history}')
print(f'Training took {training_time:.2f} seconds for {EPOCHS} epochs.')
