"""Pre-process images for training SRCNN model."""

from os import listdir, makedirs
from os.path import isfile, join, exists

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="Data input directory")
parser.add_argument("output_dir", help="Data output directory")
args = parser.parse_args()

import numpy as np
import imageio.v2 as imageio
from PIL import Image
#from scipy import misc
#from scipy.ndimage import imread

SCALE = 8.0
INPUT_SIZE = 33
LABEL_SIZE = 21
STRIDE = 14
PAD = int((INPUT_SIZE - LABEL_SIZE) / 2)

if not exists(args.output_dir):
    makedirs(args.output_dir)
if not exists(join(args.output_dir, "input")):
    makedirs(join(args.output_dir, "input"))
if not exists(join(args.output_dir, "label")):
    makedirs(join(args.output_dir, "label"))

count = 1
for f in listdir(args.input_dir):
    f = join(args.input_dir, f)
    if not isfile(f):
        continue
    
    #im = misc.imread(f, flatten=False, mode='RGB')
    #im = imageio.imread(f, as_gray=False, pilmode="RGB")
    im = np.asarray(imageio.imread(f, as_gray=False, pilmode="RGB"))
    
    w, h, c = im.shape
    w = int(w - w % SCALE)
    h = int(h - h % SCALE)
    im = im[0:w, 0:h]
    
    im_1 = Image.fromarray(im)
    size_1 = tuple((np.array(im_1.size) * 1.0/SCALE).astype(int))
    scaled = np.array(Image.fromarray(im).resize(size_1,Image.BICUBIC))
    
    scaled_1 = Image.fromarray(scaled)
    size_2 = tuple((np.array(scaled_1.size) * SCALE/1.0).astype(int))
    scaled = np.array(Image.fromarray(scaled).resize(size_2,Image.BICUBIC))
    #scaled = misc.imresize(im, 1.0/SCALE, 'bicubic')
    #scaled = misc.imresize(scaled, SCALE/1.0, 'bicubic')

    for i in range(0, h - INPUT_SIZE + 1, STRIDE):
        for j in range(0, w - INPUT_SIZE + 1, STRIDE):
            sub_img = scaled[j : j + INPUT_SIZE, i : i + INPUT_SIZE, :]
            sub_img_label = im[j + PAD : j + PAD + LABEL_SIZE, i + PAD : i + PAD + LABEL_SIZE,:]
            imageio.imwrite(join(args.output_dir, "input", str(count) + '.jpg'), sub_img)
            imageio.imwrite(join(args.output_dir, "label", str(count) + '.jpg'), sub_img_label)

            count += 1
