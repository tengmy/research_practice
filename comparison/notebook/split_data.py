import os
import numpy as np
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

INPUT_SIZE = 256
SET = [0.8]

for TRAIN_SIZE in SET:

    # Set input directory
    data_dir = "../data/color/80_20/train/"
    store_dir = "../data/color/"
    # Get classes
    classes = os.listdir(data_dir)
    #classes = os.listdir(data_dir + 'original/')
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")

    # Create train/validation directories
    #name = str(int(TRAIN_SIZE * 100)) + '_' + str(100-int(TRAIN_SIZE* 100))
    name = "64_20"
    if not os.path.exists(store_dir + name):
        os.makedirs(store_dir + name)

    train_dir = store_dir + name + '/train/'
    val_dir = store_dir + name + '/test/' 
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        os.makedirs(val_dir)

    # Create new class directories for train and validation images
    for c in classes:
        os.mkdir(train_dir + c)
        os.mkdir(val_dir + c)

    # Get path and label for each image
    db = []
    for label, class_name in enumerate(classes):
        #path = data_dir + 'original/' + class_name
        path = data_dir + class_name
        for file in os.listdir(path):
            if '.ini' not in file:
                db.append(['{}/{}'.format(class_name, file), label, class_name])
    db = pd.DataFrame(db, columns=['file', 'label', 'class_name'])
    num_images = len(db)
    print(f"Number of images: {num_images}")

    # Sample train/test observations
    np.random.seed(87)
    msk = np.random.binomial(1, TRAIN_SIZE, num_images)

    # Import images
    i = 0
    for file in tqdm(db['file'].values):

        #from_path = data_dir + 'original/' + file
        from_path = data_dir + file
        if msk[i] == 1:
            to_path = train_dir + file
        else:
            to_path = val_dir + file

        copyfile(from_path, to_path)
        i += 1
