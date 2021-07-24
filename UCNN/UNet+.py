import os
import warnings
warnings.filterwarnings("ignore")
import datetime
import glob
import random
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage.io                                    
import skimage.transform                             
from skimage.morphology import label                 
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import add, concatenate
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import Callback, TensorBoard
from IPython.display import clear_output


class PlotLosses(Callback):
    def __init__(self, figsize=None):
        super(PlotLosses, self).__init__()
        self.figsize = figsize

    def on_train_begin(self, logs={}):

        self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs.copy())

        clear_output(wait=True)
        plt.figure(figsize=self.figsize)

        for metric_id, metric in enumerate(self.base_metrics):
            plt.subplot(1, len(self.base_metrics), metric_id + 1)

            plt.plot(range(1, len(self.logs) + 1),
                     [log[metric] for log in self.logs],
                     label="training")
            if self.params['do_validation']:
                plt.plot(range(1, len(self.logs) + 1),
                         [log['val_' + metric] for log in self.logs],
                         label="validation")
            plt.title(translate_metric(metric))
            plt.xlabel('epoch')
            plt.legend(loc='center left')

        plt.tight_layout()
        plt.show();


seed = 42
random.seed = seed
np.random.seed(seed=seed)

im_height = 128
im_width = 128

def get_data(path, train=True):
    ids = next(os.walk(path + "images"))[2]

    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):  

        img = load_img(path + '/images/' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)

        fname, extension = os.path.splitext(id_)
        if extension != '.tif':
            mask_id_ = fname + '.tif'
        else:
            mask_id_ = id_
        if train:
            mask = img_to_array(load_img(path + '/labels/' + mask_id_, grayscale=True))
            # mask = img_to_array(load_img(path + '/masks/' + id_, color_mode = "grayscale"))
            mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X

def keras_model(img_width=128, img_height=128):
    n_ch_exps = [4, 5, 6, 7, 8, 9]  
    k_size = (3, 3) 
    k_init = 'he_normal' 

    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (1, img_width, img_height)
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
        input_shape = (img_width, img_height, 1)

    inp = Input(shape=input_shape)
    encodeds = []

    enc = inp
    print(n_ch_exps)
    for l_idx, n_ch in enumerate(n_ch_exps):
        enc = Conv2D(filters=2 ** n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        enc = Dropout(0.1 * l_idx, )(enc)
        enc = Conv2D(filters=2 ** n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        encodeds.append(enc)
        if n_ch < n_ch_exps[-1]: 
            enc = MaxPooling2D(pool_size=(2, 2))(enc)

    dec = enc
    print(n_ch_exps[::-1][1:])
    decoder_n_chs = n_ch_exps[::-1][1:]
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
        dec = Conv2DTranspose(filters=2 ** n_ch, kernel_size=k_size, strides=(2, 2), activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
        dec = Conv2D(filters=2 ** n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Dropout(0.1 * l_idx)(dec)
        dec = Conv2D(filters=2 ** n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)

    outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(dec)

    model = Model(inputs=[inp], outputs=[outp])

    return model

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


optimizer = 'adam'
loss = bce_dice_loss
metrics = ['accuracy']


model = keras_model(img_width=128, img_height=128)


model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

callbacks = [
     EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    ModelCheckpoint('unet+.h5', monitor='val_accuracy', mode = 'max',verbose=1, save_best_only=True, save_weights_only=True)
]

train = False
if train:
  path = "test set"
  X, y = get_data(path=path, train=True)
  X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)
  results = model.fit(X_train, y_train, batch_size=32, epochs=30, callbacks=callbacks,
                      validation_data=(X_valid, y_valid))