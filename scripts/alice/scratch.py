# -*- coding: utf-8 -*-


# For logging and loading files
import os
import pandas as pd
import numpy as np
import pickle

# ML LIbs
from sklearn import preprocessing  
from sklearn import model_selection
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
import keras


# Lag1 for PHQ
train_x = pd.read_pickle('data/pickles/day_train_phq_lag1.pickle')+1
train_x = np.reshape(train_x[:,0], (train_x.shape[0], 1, 1)  )
train_y = np.float32(pd.read_pickle('data/pickles/day_train_phq.pickle') + 1)
train_s = pd.read_pickle('data/pickles/day_train_subs.pickle')
train_subs_fac = preprocessing.LabelEncoder().fit_transform(train_s)
train_subs_norm = preprocessing.scale(train_subs_fac)


# PHQ Model
def daily_model(x):
    # Input layer
    input = Input(shape=(x.shape[1], x.shape[2]), name="feature_input")  
    lstm_layers = input
    # LSTM Layer 1: Includes DP and Regulaization
    lstm_layers = LSTM(units=64, kernel_regularizer=tf.keras.regularizers.l2(0), dropout=0, return_sequences=False )(lstm_layers)
    # Final activation layers
    output = Dense(units=1, activation = "exponential")(lstm_layers)
    # Create the model 
    model = Model(inputs=input, outputs=output)
    # Compile the model
    model.compile(
        loss='huber',
        optimizer= tf.keras.optimizers.Adam(0.005), 
        metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error']
    )
    return model


# Perform 10-fold cross-validation grouped by participant ID
validation_holdout = model_selection.GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=666)
fold_no = 1
final_df = pd.DataFrame({"fold": [], "sub_id":[], "model":[],"source":[], "predicted":[], "actual":[]})
for train_index, val_index in validation_holdout.split(train_x, train_y, groups=train_subs_fac):
 	# Create the fold for validation tuning
	x_train_fold, x_val_fold = train_x[train_index], train_x[val_index]
	y_train_fold, y_val_fold = train_y[train_index], train_y[val_index]    
	subs_train_fold, subs_val_fold = train_s[train_index], train_s[val_index]
 	# Build the model
	model = daily_model(x_train_fold)
	optional_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1000, restore_best_weights=True)
 	# Run
	history = model.fit(x_train_fold, 
					y_train_fold,	
					epochs=5000, 
					batch_size = 64,
					validation_data  = (x_val_fold, y_val_fold),
					callbacks = [optional_stopping], 	
					verbose = 2)
	# Save the model and history as pickle files
	predicted = model.predict(x_val_fold)
	actual = y_val_fold
	# Merge into a DataFrame
	results_df = pd.DataFrame({
		'fold': [fold_no] * len(predicted),
		'sub_id': subs_val_fold[:,0],
		'model': [models] * len(predicted),
		'source': [source] * len(predicted),
		'predicted': predicted[:,0],
		'actual': actual[:,0]
	})
 
