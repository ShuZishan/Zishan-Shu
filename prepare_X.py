# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 10:02:04 2022

@author: pku
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.applications.xception import preprocess_input
import matplotlib.pyplot as plt 
import h5py
#%%
img_dir = './Pics/'
val_img_dir = './valPics/'
nb_train_samples = 16000
nb_val_samples = 500
im_size = 165

imgs = []

for i in range(nb_train_samples):
    img = Image.open(os.path.join(img_dir,'%d' % (i+1) + '.bmp'))
    img = img.resize((im_size,im_size),Image.BICUBIC)
    img1 = np.array(img)
    imgs.append(img1)

imgs = np.stack(imgs)


val_imgs = []

for i in range(nb_val_samples):
    img = Image.open(os.path.join(val_img_dir,'%d' % (i+1) + '.bmp'))
    img = img.resize((im_size,im_size),Image.BICUBIC)
    img1 = np.array(img)
    val_imgs.append(img1)

val_imgs = np.stack(val_imgs)
#%%
train_x = preprocess_input(imgs)
val_x = preprocess_input(val_imgs)
#%%

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


input_tensor = Input(shape=(im_size, im_size, 3))
base = applications.Xception(input_tensor=input_tensor, weights='imagenet', include_top=True)
base.summary()
#%%
x = base.layers[77].output
model = Model(inputs = base.input, outputs = x)
model.summary()
#%%
train_fea = model.predict(train_x)
val_fea = model.predict(val_x)

np.save('train_x77.npy',train_fea)
np.save('val_x77.npy',val_fea)