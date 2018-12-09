'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import tensorflow as tf
from keras.utils import multi_gpu_model

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model





from keras import backend as K
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.get_keras_data as gkd

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.models import load_model

import os
import pickle
import numpy as np


import get_lenet as gln

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.get_keras_data as gkd



def run_config(z, x):
  total_train_size = 50000
  total_base_size = 5000
  train_size = int(total_base_size + \
    (total_train_size - total_base_size)*z*1.0)
  (nf_conv1, nf_conv2, nf_dense1, nf_dense2,
    is_reg1, is_reg2, lr, momentum, batch_size) = x

  epochs = 10

  if(nf_conv1 > nf_conv2):
    nf_conv1 = nf_conv2

  if(nf_dense2 > nf_dense1):
    nf_dense2 = nf_dense1

  if(is_reg1 > 0.5):
    is_reg1 = True
  else:
    is_reg1 = False

  if(is_reg2 > 0.5):
    is_reg2 = True
  else:
    is_reg2 = False
  
  return train_config(nf_conv1, nf_conv2, nf_dense1, nf_dense2,
    is_reg1, is_reg2, lr, momentum, batch_size, epochs, train_size)

def train_config(nf_conv1, nf_conv2, nf_dense1, nf_dense2,
    is_reg1, is_reg2, lr, momentum, batch_size, epochs, train_size):
  
  x_train, y_train, x_test, y_test = \
    gkd.get_data("mnist")
  x_train = x_train[:train_size]
  y_train = y_train[:train_size]

  x_train /= 255
  x_test /= 255

  num_classes = int(y_train.shape[1])

  weight_path="temp.hdf5"
  checkpoint = ModelCheckpoint(weight_path, monitor='val_acc', \
    verbose=0, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]

  with tf.device('/gpu:0'):
    model = gln.get_conv_net_conf(x_train.shape[1:], 
                  num_classes, nf_conv1, nf_conv2, 
                  nf_dense1, nf_dense2, is_reg1, is_reg2,
                  lr, momentum)
  # parallel_model = multi_gpu_model(model, gpus=4)
  parallel_model = model
  print('Not using data augmentation.')
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True,
            callbacks=callbacks_list, verbose=1)

  with tf.device("/gpu:0"):
    # load the weights and model from .h5
    model = load_model(weight_path)
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  return score

if __name__ == '__main__':
  nf_conv1 = 6
  nf_conv2 = 16
  nf_dense1 = 120
  nf_dense2 = 84
  is_reg1 = True
  is_reg2 = True
  batch_size = 128
  epochs = 10
  lr = 0.1
  momentum = 0.9
  # train_config(nf_conv1, nf_conv2, nf_dense1, nf_dense2,
  #     is_reg1, is_reg2, lr, momentum, batch_size, epochs)

  nf_conv1 = 6
  nf_conv2 = 16
  nf_dense1 = 120
  nf_dense2 = 84
  is_reg1 = False
  is_reg2 = False
  batch_size = 128
  epochs = 10
  lr = 0.1
  momentum = 0.9
  # train_config(nf_conv1, nf_conv2, nf_dense1, nf_dense2,
  #     is_reg1, is_reg2, lr, momentum, batch_size, epochs)

  x = (nf_conv1, nf_conv2, nf_dense1, nf_dense2,
      is_reg1, is_reg2, lr, momentum, batch_size)
  z = 0.1
  run_config(z, x)

