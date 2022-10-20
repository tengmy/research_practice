"""Train SRCNN model."""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
from os.path import join, exists
from os import makedirs,listdir

import numpy as np
import imageio.v2 as imageio
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Input, Convolution2D
#from tensorflow.keras.layers import Input,Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam 

#tf.debugging.set_log_device_placement(True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

EXPORT_DIR = "../../models/"
if not exists(EXPORT_DIR):
    makedirs(EXPORT_DIR)

DATA_DIR = "../../data/color/srcnn_input/"

input_dir = join(DATA_DIR, "input")
label_dir = join(DATA_DIR, "label")

if not (exists(input_dir) and exists(label_dir)):
    print("Training data not found")
    sys.exit(1)

# Specify model
inputs = Input(shape=(33, 33, 3))
x = Convolution2D(64, (9, 9), activation='relu')(inputs)
x = Convolution2D(32, (1, 1), activation='relu')(x)
x = Convolution2D(3, (5, 5))(x)

#model = tf.keras.Model(inputs=inputs, outputs=x)
model = Model(inputs =inputs, outputs =x)

model.compile(optimizer=Adam(lr=0.001), loss='mse',metrics=['accuracy'])#metrics=['accuracy']

#X_train = np.array([misc.imread(join(input_dir, f))[:,:,:] for f in os.listdir(input_dir)])
#y_train = np.array([misc.imread(join(label_dir, f))[:,:,:] for f in os.listdir(label_dir)])
X_train = np.array([np.asarray(imageio.imread(join(input_dir, f)))[:,:,:] for f in listdir(input_dir)])
y_train = np.array([np.asarray(imageio.imread(join(label_dir, f)))[:,:,:] for f in listdir(input_dir)])

# Set up callbacks
model_name = "SRCNN_PV_2"
filepath = EXPORT_DIR + model_name + ".hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_accuracy',
                             mode='auto',
                             verbose=1,
                             save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                              factor=0.5,
                              patience=5,
                              cooldown=5)

filepath = EXPORT_DIR + model_name + ".csv"
csv_logger = CSVLogger(filepath)
callbacks_list = [checkpoint, reduce_lr, csv_logger]

model.fit(X_train, y_train, 
          batch_size=128, 
          epochs=500,
          validation_split=0.2,
          callbacks=callbacks_list)



