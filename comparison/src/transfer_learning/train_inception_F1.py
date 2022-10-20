"""Train CNN model based on InceptionV3 network."""

#import keras
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.optimizers import Adam

import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

a = argparse.ArgumentParser()
a.add_argument("image_type")
a.add_argument("train_split")
args = a.parse_args()

IM_WIDTH, IM_HEIGHT = 299, 299
NB_EPOCHS = 30
BATCH_SIZE = 32
FC_SIZE = 1024

DATA_DIR = "../../data/" + str(args.image_type) + "/"  + str(args.train_split)
TRAIN_DIR = DATA_DIR + "/train/"
VAL_DIR = DATA_DIR + "/validation/"
EXPORT_DIR = "../../models/"

# Get image classes
classes = os.listdir(TRAIN_DIR)
num_classes = len(classes)

# Get path and label for each image
db = []
for label, class_name in enumerate(classes):

    # Train
    path = TRAIN_DIR + class_name
    for file in os.listdir(path):
        db.append(['{}/{}'.format(class_name, file), label, class_name, 1])

    # Validation
    path = VAL_DIR + class_name
    for file in os.listdir(path):
        db.append(['{}/{}'.format(class_name, file), label, class_name, 0])

db = pd.DataFrame(db, columns=['file', 'label', 'class_name', 'train_ind'])

num_train_samples = db.train_ind.sum()
num_val_samples = len(db) - num_train_samples

# Specify data generator inputs
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=BATCH_SIZE,
)
validation_generator = test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=BATCH_SIZE,
)

# Get Inception model without final layer
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add new fully connected layer to base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(FC_SIZE, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

#add new scores
import tensorflow_addons as tfa
f1 = tfa.metrics.F1Score(num_classes=38, average="micro")
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()



# Compile model
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy',precision,recall,f1])

# Set up callbacks
model_name = "inception_" + str(args.image_type) + "_" + str(args.train_split)
filepath = EXPORT_DIR + model_name + ".hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_accuracy',
                             mode='max',
                             verbose=1,
                             save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                              factor=0.2,
                              patience=5,
                              cooldown=5)

filepath = EXPORT_DIR + model_name + ".csv"
csv_logger = CSVLogger(filepath)
callbacks_list = [checkpoint, reduce_lr, csv_logger]

# Fit transfer learning model
history = model.fit(
    train_generator,
    epochs=NB_EPOCHS,
    steps_per_epoch=500, #num_train_samples / BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=500, #num_val_samples / BATCH_SIZE,
    callbacks=callbacks_list)

