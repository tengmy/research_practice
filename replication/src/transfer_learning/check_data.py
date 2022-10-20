"""Train CNN model based on InceptionV3 network."""

#import keras
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
NB_EPOCHS = 20
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

print(num_train_samples)
print(num_val_samples)
