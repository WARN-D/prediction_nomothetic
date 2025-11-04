# -*- coding: utf-8 -*-
"""
WARN-D Keras Tuning Script
"""

# For logging and loading files
import os
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import sys
import argparse

# ML Libs
from sklearn import preprocessing  
from sklearn import model_selection
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, TimeDistributed
from tensorflow.keras.models import Model
import keras
import keras_tuner  


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--day_week", type=str, required=True)
parser.add_argument("--var", type=str, choices=["ema", "epa", "all"], required=True)
args = parser.parse_args()
# Get arguments
day_week = args.day_week
var = args.var

# Ensure GPUs are visible
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Model function
def build_model(hp, x):

    # Input layer
    feature_input = Input(shape=(x.shape[1], x.shape[2]), name="feature_input")

    # Number of LSTM layers to use
    num_layers = hp.Int("num_lstm_layers", min_value=1, max_value=4, step=1)
    feature_model = feature_input
    for i in range(num_layers):
        units = int(hp.Int(f"units_{i}", min_value=8, max_value=128, step=2, sampling='log'))
        dropout = hp.Float(f"dp_{i}", min_value=0.0, max_value=0.5, step=0.1)
        l2 = hp.Float(f"l2_{i}", min_value=0.0, max_value=0.5, step=0.1)
        return_seq = i < num_layers - 1

        lstm_layer = LSTM(units=units,
                          dropout=dropout,
                          kernel_regularizer=tf.keras.regularizers.l2(l2),
                          return_sequences=return_seq)

        feature_model = lstm_layer(feature_model)

    # Output layer activation based on task type
    output = Dense(units=1 ,activation = hp.Choice('activation_0', values=['relu', 'exponential', 'sigmoid']))(feature_model)

    model = Model(inputs=feature_input, outputs=output)

    # Compile
    learning_rate = hp.Float("learning_rate", min_value=0.0005, max_value=0.1, step=0.0005)
    model.compile(
        loss='huber',
        optimizer= tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0), 
        metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error']
    )
    return model


# Load data
train_x = pd.read_pickle(f'data/pickles/{day_week}_train_{var}.pickle')
train_y = np.float32(pd.read_pickle(f'data/pickles/{day_week}_train_phq.pickle') + 1)
train_s = pd.read_pickle(f'data/pickles/{day_week}_train_subs.pickle')
train_subs_fac = preprocessing.LabelEncoder().fit_transform(train_s)


# Set up the HP search
tuner = keras_tuner.Hyperband(
    hypermodel = lambda hp: build_model(hp, train_x),
    max_epochs = 10000,
    executions_per_trial = 2,
    seed = 420,
    objective = "val_loss",
    overwrite = False, 
    directory = "/home/toutounjir/data1/project_nomothetic",
    project_name = f"results/hb_{day_week}_{var}"
    )


# Set callbacks
log_path = Path(f"/home/toutounjir/data1/project_nomothetic/logs/hb_{day_week}_{var}/fit")
log_dir = log_path / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1, update_freq='epoch' )
backups = keras.callbacks.BackupAndRestore(backup_dir=(f"/home/toutounjir/data1/project_nomothetic/backup/hb_{day_week}_{var}"))
optional_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=1000, restore_best_weights=True)

# Perform 10-fold cross-validation grouped by participant ID
validation_holdout = model_selection.GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=666)
fold_no = 1

for train_index, val_index in validation_holdout.split(train_x, train_y, groups=train_subs_fac):
    
    # Create the fold for validation tuning
    x_train_fold, x_val_fold = train_x[train_index], train_x[val_index]
    y_train_fold, y_val_fold = train_y[train_index], train_y[val_index]    
    
    # Build the model with parallel strategy
    tuner.search(x_train_fold,
                y_train_fold,
                epochs=10000, 
                batch_size = 64,
                validation_data  = (x_val_fold, y_val_fold),
                callbacks = [optional_stopping, backups, tensorboard_callback], 
                verbose = 2)

