from tqdm import tqdm # Cool progress bar
import numpy as np
import pandas as pd
import cv2
import os
from tensorflow.keras.layers import Conv2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.activations import relu as Relu
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import h5py

################################# "FUNCTIONAL" INPUTS ################################# START
## Input
USER_NAME       = 'dat300-h24-3'
PATH_TO_DATASET = f'/mnt/users/{USER_NAME}/tree_train.h5'
MODEL_NAME      = 'VGG16_model_dat300'

# Parameters
BATCH_SIZE = 128
EPOCHS = 20

## Outputs
# Model file
PATH_TO_STORE_MODEL = f'/mnt/users/{USER_NAME}/models'

def F1_score(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    P = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (P + K.epsilon())

    Pred_P = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = TP / (Pred_P + K.epsilon())
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def rescale_images(images):
    """Rescales images to the range [0, 1]."""
    if np.max(images) > 1:
      return images / np.max(images)
    else:
      return images

def get_data():
    with h5py.File(f"{PATH_TO_DATASET}",'r') as f:
        print('Datasets in file:', list(f.keys()))
        X_train = np.asarray(f['X'])
        y_train = np.asarray(f['y'],dtype=np.float64) # avoid clashing dtype errors
    
    return rescale_images(X_train), y_train

# Joining paths
path_to_store_training_history = os.path.join(PATH_TO_STORE_MODEL, MODEL_NAME + '_training_history.png')
path_to_store_model = os.path.join(PATH_TO_STORE_MODEL, MODEL_NAME + '.keras')

################################# VISUALIZATION ################################# START

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
    plt.savefig(path_to_store_training_history)
    plt.close()

################################# VISUALIZATION ################################# END

class UNetWithPretrainedEncoder:

    def __init__(self, input_shape=(128, 128, 3), dropout_rate=0.3, learning_rate=0.001, activation='relu'):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation = activation

    def build_model(self):
        # Load the pretrained VGG16 model as the encoder, with weights pre-trained on ImageNet.
        encoder = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Capture the encoder layers we’ll use as skip connections.
        inputs = encoder.input
        skip1 = encoder.get_layer("block1_conv2").output
        skip2 = encoder.get_layer("block2_conv2").output
        skip3 = encoder.get_layer("block3_conv3").output
        skip4 = encoder.get_layer("block4_conv3").output
        bottleneck = encoder.get_layer("block5_conv3").output

        # Decoder Block 1
        u4 = UpSampling2D((2, 2))(bottleneck)
        u4 = concatenate([u4, skip4])
        c4 = Conv2D(1024, (3, 3), padding='same')(u4)  # Increased filters
        c4 = self._apply_activation(c4)
        c4 = BatchNormalization()(c4)
        c4 = Dropout(self.dropout_rate)(c4)  # Add dropout
        c4 = Conv2D(1024, (3, 3), padding='same')(c4)
        c4 = self._apply_activation(c4)
        c4 = BatchNormalization()(c4)

        # Decoder Block 2
        u3 = UpSampling2D((2, 2))(c4)
        u3 = concatenate([u3, skip3])
        c3 = Conv2D(512, (3, 3), padding='same')(u3)
        c3 = self._apply_activation(c3)
        c3 = BatchNormalization()(c3)
        c3 = Dropout(self.dropout_rate)(c3)  # Add dropout
        c3 = Conv2D(512, (3, 3), padding='same')(c3)
        c3 = self._apply_activation(c3)
        c3 = BatchNormalization()(c3)

        # Decoder Block 3
        u2 = UpSampling2D((2, 2))(c3)
        u2 = concatenate([u2, skip2])
        c2 = Conv2D(256, (3, 3), padding='same')(u2)
        c2 = self._apply_activation(c2)
        c2 = BatchNormalization()(c2)
        c2 = Dropout(self.dropout_rate)(c2)  # Add dropout
        c2 = Conv2D(256, (3, 3), padding='same')(c2)
        c2 = self._apply_activation(c2)
        c2 = BatchNormalization()(c2)

        # Decoder Block 4
        u1 = UpSampling2D((2, 2))(c2)
        u1 = concatenate([u1, skip1])
        c1 = Conv2D(128, (3, 3), padding='same')(u1)
        c1 = self._apply_activation(c1)
        c1 = BatchNormalization()(c1)
        c1 = Dropout(self.dropout_rate)(c1)  # Add dropout
        c1 = Conv2D(128, (3, 3), padding='same')(c1)
        c1 = self._apply_activation(c1)
        c1 = BatchNormalization()(c1)

        # Output Layer
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c1)

        # Compile Model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="binary_crossentropy", metrics=[
            F1_score, 'accuracy', TruePositives(), TrueNegatives(), FalsePositives(), FalseNegatives()
        ])
        return model

    def _apply_activation(self, x):
        return Relu(x, max_value=6)

    def train_model(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
        model = self.build_model()
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size, epochs=epochs,
            callbacks=[early_stopping, model_checkpoint, reduce_lr]
        )
        return model, history
    
################################# Output ################################# START

X_train, y_train = get_data()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model, history = UNetWithPretrainedEncoder().train_model(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Save model and plot training history
model.save(path_to_store_model)
plot_training_history(history)