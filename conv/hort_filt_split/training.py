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

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.nasnet import NASNetMobile



from keras import backend as K
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# print(os.path.dirname(os.path.abspath(__file__)))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.get_keras_data as gkd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.models import load_model

import os
import pickle
import numpy as np


import conv.networks.gen_conv_net as gcn
# import get_data as gd
import conv.networks.get_vgg16_cifar10 as gvc
import conv.networks.get_wide_res_networks as gwrn

network = sys.argv[1] #vgg, wrn, conv
data_set = sys.argv[2] #mnist, cifar10, cifar100

batch_size = 128 #32
# num_classes = 100 # 10 for cifar10 # 100 for cifar100
epochs = 200
data_augmentation = True
# num_predictions = 20
save_dir = os.path.join(os.getcwd(), '../saved_models')
model_name = network+'_'+data_set+'_'


# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

for resize_factor in [0]:#,1,2]:
  # Do not resize input and check the accuracy
  x_train, y_train, x_test, y_test = \
    gkd.get_data(data_set)

  # x_train, x_test = gd.scale_image(x_train, x_test)
  x_train /= 255
  x_test /= 255
  num_classes = int(y_train.shape[1])


  print(x_train.shape, y_train.shape, \
    x_test.shape, y_test.shape)

  # with tf.device('/cpu:0'):
  with tf.device('/gpu:0'):
    if(network == "vgg"):
      # model = gvc.get_conv_net_v1(x_train.shape[1:], \
      #   num_classes, 3-resize_factor)
      model = gvc.get_conv_net_cifar100(x_train.shape[1:], \
        num_classes, 3-resize_factor)
    elif(network == "wrn"):
      model = gwrn.create_wide_residual_network(x_train.shape[1:], 
        2-resize_factor,
        nb_classes=num_classes, N=4, k=8, dropout=0.3)
    elif(network == "conv"):
      model = gcn.get_conv_net(x_train.shape[1:], \
        num_classes, 2-resize_factor)
    else:
      print("Invlaid Network Type")
      exit(0)


    # model = gln.get_conv_net(x_train.shape[1:], num_classes) 

    print("Input Shape ", x_train.shape[1:])    

  # parallel_model = multi_gpu_model(model, gpus=4)
  parallel_model = model

  if not data_augmentation:
      print('Not using data augmentation.')
      model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True)
  else:
      print('Using real-time data augmentation.')
      # This will do preprocessing and realtime data augmentation:
      datagen = ImageDataGenerator(
          featurewise_center=False,  # set input mean to 0 over the dataset
          samplewise_center=False,  # set each sample mean to 0
          featurewise_std_normalization=False,  # divide inputs by std of the dataset
          samplewise_std_normalization=False,  # divide each input by its std
          zca_whitening=False,  # apply ZCA whitening
          rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
          width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
          height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
          horizontal_flip=True,  # randomly flip images
          vertical_flip=False)  # randomly flip images

      # Compute quantities required for feature-wise normalization
      # (std, mean, and principal components if ZCA whitening is applied).
      datagen.fit(x_train)

      weight_path = os.path.join(save_dir, \
        model_name+str(resize_factor)+".h5")
      # model.save(weight_path)
      
      modelCheckpoint = ModelCheckpoint(weight_path, 
        monitor='loss', verbose=0, save_best_only=True, 
        save_weights_only=False, mode='auto', period=1)

      callbacks = [
                  modelCheckpoint
                  #   earlyStopping, 
                  #   reduceonplateau,
                  #   csv_logger
                  ]

      # initiate RMSprop optimizer
      optimizers = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
      # optimizers = keras.optimizers.Adam(lr=0.01, decay=1e-6)

      parallel_model.compile(loss='categorical_crossentropy', \
        optimizer=optimizers, metrics=['accuracy'])

      parallel_model.fit_generator(datagen.flow(x_train, y_train,
                                       batch_size=batch_size),
                          steps_per_epoch=x_train.shape[0] // batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test),
                          callbacks = callbacks)


      # lrf = 0.1
      # lr_decay = 1e-6
      # sgd = optimizers.SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
      # parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

      # for epoch in range(1,epochs):
      #   if epoch%25==0 and epoch>0:
      #     lrf/=2
      #     sgd = optimizers.SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
      #     parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

      #   # Fit the model on the batches generated by datagen.flow().
      #   # https://keras.io/models/sequential/ - fit_generator
      #   # https://keras.io/preprocessing/image/ - datagenerator
      #   parallel_model.fit_generator(datagen.flow(x_train, y_train,
      #                                  batch_size=batch_size),
      #                     steps_per_epoch=x_train.shape[0] // batch_size,
      #                     epochs=epoch,
      #                     validation_data=(x_test, y_test),
      #                     callbacks = callbacks,
      #                     initial_epoch=epoch-1)

  # Not required since model and weight are stored together
  # model_path = os.path.join(save_dir, \
  #   "keras_cifar10_model_"+str(resize_factor)+".h5")
  # with open(model_path, "w") as text_file:
  #   text_file.write(model.to_json())
  print('Saved trained model and weights at %s ' % weight_path)

