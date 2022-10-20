"""Generate images using trained SRCNN model."""

#import tensorflow as tf
from tensorflow.keras.models import load_model

import os
from os import listdir, makedirs
from os.path import isfile, join, basename, exists
from skimage.util import img_as_ubyte

import numpy as np
import pandas as pd

#from scipy import misc
import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm
import cv2

DATA_DIR = "../../data/color/"
MODEL_DIR = "../../models/SRCNN_PV_2.hdf5"
TRAIN_SIZE = 0.7   
CHUNK_SIZE = 1000

# Scaling params
SCALE = 8
INPUT_SIZE = 33
LABEL_SIZE = 21
PAD = int((INPUT_SIZE - LABEL_SIZE) / 2)

# Get classes
classes = listdir(DATA_DIR + 'original/') #species_disease
if '.DS_Store' in classes:  # Added by BR on 4/19 to enhance compatibility with Mac
    classes.remove('.DS_Store')
num_classes = len(classes)

# Create train/validation directories
name = str(int(TRAIN_SIZE * 100)) + '_' +  str(int((1-TRAIN_SIZE) * 100)) 

srcnn_name = name + "_srcnn_v2"
bicubic_name = name + "_bicubic_v2"

if not os.path.exists(DATA_DIR + srcnn_name):
    os.mkdir(DATA_DIR + srcnn_name)
    os.mkdir(DATA_DIR + bicubic_name)

srcnn_train_dir = DATA_DIR + srcnn_name + '/train/'
bicubic_train_dir = DATA_DIR + bicubic_name + '/train/'

if not os.path.exists(srcnn_train_dir):
    os.mkdir(srcnn_train_dir)
    os.mkdir(bicubic_train_dir)

srcnn_val_dir = DATA_DIR + srcnn_name + '/validation/'
bicubic_val_dir = DATA_DIR + bicubic_name + '/validation/'

if not os.path.exists(srcnn_val_dir):
    os.mkdir(srcnn_val_dir)
    os.mkdir(bicubic_val_dir)

# Create new class directories for train and validation images
for c in classes:
    if not os.path.exists(srcnn_train_dir + c):
        os.mkdir(srcnn_train_dir + c)
        os.mkdir(srcnn_val_dir + c)
        os.mkdir(bicubic_train_dir + c)
        os.mkdir(bicubic_val_dir + c)

# Get path and label for each image
db = []
for label, class_name in enumerate(classes):
    path = DATA_DIR + 'original/' + class_name
    for file in os.listdir(path):
        if '.ini' not in file:
            db.append(['{}/{}'.format(class_name, file), label, class_name])
db = pd.DataFrame(db, columns=['file', 'label', 'class_name'])
num_images = len(db)
print(f"Number of images: {num_images}")

# Sample train/test observations
np.random.seed(87)
msk = np.random.binomial(1, TRAIN_SIZE, num_images)

# Load SRCNN model
model = load_model(MODEL_DIR)

chunk_num = 0
count = 0
figure_num=0
while count < num_images:

    temp = []
    cstart = chunk_num * CHUNK_SIZE
    cend = min((chunk_num + 1) * CHUNK_SIZE, num_images)
    chunk_num += 1

    # Read images
    print(f"Chunk number: {chunk_num}")
    print("Importing images...")
    for f in tqdm(db['file'].values[cstart:cend]):

        count += 1
        from_path = DATA_DIR + 'original/' + f
        #im = misc.imread(from_path, flatten=False, mode='RGB')
        #im = cv2.imread()
        im = np.asarray(imageio.imread(from_path, as_gray=False, pilmode="RGB"))

        w, h, c = im.shape
        w = w - int(w % SCALE)
        h = h - int(h % SCALE)
        im = im[0:w, 0:h, :]
        
        #scaled = misc.imresize(im, 1.0/SCALE, 'bicubic')
        #scaled = misc.imresize(scaled, SCALE/1.0, 'bicubic')
        scaled = np.array(Image.fromarray(im).resize((int(w*1.0/SCALE),int(h*1.0/SCALE)),Image.BICUBIC))
        
        w_now, h_now = scaled.shape[:2]
        scaled = np.array(Image.fromarray(scaled).resize((int(w_now*SCALE/1.0),int(h_now*SCALE/1.0)),Image.BICUBIC))
        
        #im = Image.fromarray(im)
        #size_1 = tuple((np.array(im.size) * 1.0/SCALE).astype(int))
        #scaled = np.array(im.resize(size_1,Image.BICUBIC))
                        
        #scaled = Image.fromarray(scaled)
        #size_2 = tuple((np.array(scaled.size) * SCALE/1.0).astype(int))
        #scaled = np.array(scaled.resize(size_2,Image.BICUBIC))

        temp.append(scaled)

    temp = np.array(temp)
    #print(temp.shape)
    im_new = np.zeros(temp.shape)
    im_scaled = temp[:, PAD: int(w - w % INPUT_SIZE), PAD: int(h - h % INPUT_SIZE), :]

    # Get predictions
    print("Running prediction...")
    for i in tqdm(range(0, h - INPUT_SIZE + 1, LABEL_SIZE)):
        for j in range(0, w - INPUT_SIZE + 1, LABEL_SIZE):
            
            sub_img = temp[:, j : j + INPUT_SIZE, i : i + INPUT_SIZE, :]
            prediction = model.predict(sub_img, batch_size=16, verbose=0)
            prediction = prediction.reshape(cend-cstart, LABEL_SIZE, LABEL_SIZE, 3)

            im_new[:, j + PAD : j + PAD + LABEL_SIZE, i + PAD : i + PAD + LABEL_SIZE, :] = prediction
            
    im_new = im_new[:, PAD : int(w - w % INPUT_SIZE), PAD : int(h - h % INPUT_SIZE), :]

    # Save results
    print("Saving scaled images...")
    i=0
    for f in tqdm(db['file'].values[cstart:cend]):
        if msk[figure_num] == 1:
            srcnn_to_path = srcnn_train_dir + f
            bicubic_to_path = bicubic_train_dir + f
            print("train+1")
        else:
            srcnn_to_path = srcnn_val_dir + f
            bicubic_to_path = bicubic_val_dir + f
        
        
        # Bicubic , save image to unit8
        im = im_scaled[i,:,:,:]
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        #misc.imsave(bicubic_to_path, im)
        imageio.imwrite(bicubic_to_path, im)
                        
    
        # SRCNN
        im = im_new[i,:,:,:]
        #misc.imsave(srcnn_to_path, im)
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        imageio.imwrite(srcnn_to_path, im)
        
        figure_num += 1
        i += 1
        
        


