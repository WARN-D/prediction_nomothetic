# For logging and loading files
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import argparse

# ML Libs
from sklearn import preprocessing  
from sklearn import model_selection
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, TimeDistributed
from tensorflow.keras.models import Model
import shap

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

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--day_week", type=str, required=True)
parser.add_argument("--var", type=str, choices=["ema", "epa", "all", "phq"], required=True)
args = parser.parse_args()

# get arguments
models = args.day_week
source = args.var
if source == "phq":
    source = "phq_lag1"

 # Set-up dictionary with model parameters
hyperparams = {
    "day": {
        "n_layers": 5,
        "units": [64, 16, 16, 8, 16],
        "dropouts": [0.3, 0, 0, 0, 0.2],
        "l2_regs": [0, 0, 0, 0, 0],
        "lr": 0.0005,
        "activation": "exponential"
    },
    "week": {
        "n_layers": 5,
        "units": [8, 8, 16, 32, 128],
        "dropouts": [0, 0, 0, 0, 0.4],
        "l2_regs": [0, 0, 0, 0, 0],
        "lr": 0.0005,
        "activation": "exponential"
    }
}

# pick the right params based on models argument
params = hyperparams[models] 
def final_model(x, params):
    input_layer = Input(shape=(x.shape[1], x.shape[2]))
    x_model = input_layer

    for i in range(params["n_layers"]):
        return_seq = (i < params["n_layers"] - 1)
        x_model = LSTM(
            units=params["units"][i],
            dropout=params["dropouts"][i],
            kernel_regularizer=tf.keras.regularizers.l2(params["l2_regs"][i]),
            return_sequences=return_seq
        )(x_model)

    output = Dense(units=1, activation=params["activation"])(x_model)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        loss="huber",
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]),
        metrics=["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error"]
    )
    return model

#Load training data
train_x = pd.read_pickle('data/pickles/'+ models + '_train_' + source + '_new.pickle')
train_y = np.float32(pd.read_pickle('data/pickles/'+ models + '_train_phq_new.pickle') + 1)
train_s = pd.read_pickle('data/pickles/' + models+ '_train_subs_new.pickle')
train_subs_fac = preprocessing.LabelEncoder().fit_transform(train_s)
train_subs_norm = preprocessing.scale(train_subs_fac)

# Read features names
if source == "phq_lag1":
    feature_names = ["phq_lag1"]
    train_x = np.reshape(train_x[:,0], (train_x.shape[0], 1, 1)  )
else:
	feature_names = pd.read_pickle('data/pickles/vars_' + source + '_new.pickle')


# -----------------------
# Grouped 10-fold CV
# -----------------------a

# Perform 10-fold cross-validation grouped by participant ID
validation_holdout =model_selection.GroupKFold(n_splits=10)
fold_no = 1
final_df = pd.DataFrame({"fold": [], "sub_id":[], "model":[],"source":[], "predicted":[], "actual":[]})
for train_index, val_index in validation_holdout.split(train_x, train_y, groups=train_subs_fac):
	
 	# Create the fold for validation tuning
	print(f'Creating fold {fold_no} data...')
	x_train_fold, x_val_fold = train_x[train_index], train_x[val_index]
	y_train_fold, y_val_fold = train_y[train_index], train_y[val_index]    
	subs_train_fold, subs_val_fold = train_s[train_index], train_s[val_index]
	
 	# Build the model
	print(f'Building model for fold {fold_no} ...')
	tf.keras.backend.clear_session()
	model = final_model(x_train_fold, params)
	optional_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1000, restore_best_weights=True)
	
 	# Run
	print(f'Training for fold {fold_no} ...')
	history = model.fit(x_train_fold, 
					y_train_fold,	
					epochs=5000,
					batch_size = 64,
					validation_data  = (x_val_fold, y_val_fold),
					callbacks = [optional_stopping], 	
					verbose = 0)
 
	# Save the model and history as pickle files
	predicted = model.predict(x_val_fold)[:,0].flatten()
	actual = y_val_fold

	# Merge into a DataFrame
	print(f'Saving results for fold {fold_no} ...')
	results_df = pd.DataFrame({
		'fold': [fold_no] * len(predicted),
		'sub_id': subs_val_fold[:,0],
		'model': [models] * len(predicted),
		'source': [source] * len(predicted),
		'predicted': predicted[:],
		'actual': actual[:,0]
		})
 
	# Save the results
	results_df.to_csv(f'results/results_training_{models}_{source}_fold{fold_no}.csv', index=False)
	final_df = pd.concat([final_df,results_df])
	model.save(f'results/model_{models}_{source}_fold_{fold_no}.keras')
	# Update fold number
	fold_no += 1
final_df.to_csv(f'results/final_results_{models}_{source}.csv', index=False)

