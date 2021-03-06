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

from keras import backend as K
import get_wide_res_networks as gwrn

import get_vgg16_cifar10 as gvc
import gen_conv_net as gcn

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.get_keras_data as gkd

import restructure_models as rm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.models import load_model

# List of experiments done:
# Train nets independently with different values
# Train only the last layer reusing the top layers without resizing
# Train only the last layer with resizing of the input


import os
import pickle
import numpy as np


import gen_conv_net as gcn
# import get_data as gd
# import get_vgg16_cifar10 as gvc

batch_size = 128 #32
# num_classes = 100 # 10 for cifar10 # 100 for cifar100
epochs = 200
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
# model_name = 'keras_cifar10_trained_model.h5'

# weight_path = "../../data/ml_tutorial/conv/cifar10_conv_v0/keras_cifar10_weight_0.h5"
weight_path = "../../data/ml_tutorial/conv/cifar10_conv_v0/keras_cifar10_weight_0.h5"
weight_path = "../../data/ml_tutorial/conv/mnist_conv_v1/keras_cifar10_weight_0.h5"

# weight_path = "../../data/ml_tutorial/conv/saved_model_vgg_v3/keras_cifar10_weight_0.h5"


def get_mask_weights(weight_list_copy, mask):
  for i,weight in enumerate(weight_list_copy):
    if(len(weight.shape) == 4 and i != 0):
      if(weight.shape[0] == 3 and weight.shape[1] == 3):
        for j in range(int(mask*weight.shape[2]/4), weight.shape[2]):
          weight[:,:,j,:] = 0
      if(weight.shape[0] == 1 and weight.shape[1] == 1):
        for j in range(int(mask*weight.shape[3]/4), weight.shape[3]):
          weight[:,:,:,j] = 0
      print("Layer modified ", i, int(mask*weight.shape[2]/4), weight.shape[2],
        (mask*weight.shape[3]/4), weight.shape[3])

      # if(weight.shape[0] == 3 and weight.shape[1] == 3):
      #   for j in range(weight.shape[2]-mask, weight.shape[2]):
      #     weight[:,:,j,:] = 0
      # if(weight.shape[0] == 1 and weight.shape[1] == 1):
      #   for j in range(weight.shape[3]-mask, weight.shape[3]):
      #     weight[:,:,:,j] = 0
    print("Weight shape ", i, weight.shape)
  return weight_list_copy

# Save model and weights
if not os.path.isdir(save_dir):
  os.makedirs(save_dir)

# Assume the top network has already been trained
# Now for the smaller networks we pop up some conv layers
# Retrain the weights to classifiy the images
for mask in [0.15, 0.85]:#[3]:#0.65,0.75]: # KJN Change this
  # Do not resize input and check the accuracy
  # x_train, y_train, x_test, y_test = \
  #   gd.get_cifar_data(0, num_classes)
  # x_train, y_train, x_test, y_test = \
  #   gd.get_cifar_data(0, num_classes)
  x_train, y_train, x_test, y_test = \
    gkd.get_data("mnist")
  # x_train, x_test = gkd.scale_image(x_train, x_test) # KJN Change this
  x_train /= 255
  x_test /= 255
  num_classes = int(y_train.shape[1])

  print(x_train.shape, y_train.shape, \
    x_test.shape, y_test.shape)

  with tf.device('/gpu:0'):
    model = load_model(weight_path)
    weight_list = model.get_weights()

  # model.summary()
  evaluation = model.evaluate(x_test, y_test)
  print("Evaluation Test Set ", evaluation)
  model.summary()


  new_model = gcn.get_conv_vert_net(x_train.shape[1:], num_classes, 2, 1)
  # new_model.summary()
  # exit()

  new_weight_list = new_model.get_weights()

  # 1.33 scaling 50% filter conv mnist (even 1.25, 1.1 seems to work)
  # 1.33 scaling 75% filters conv mnist
  # 2.25 (2.0) scaling 25% filters conv mnist

  scale = 2.0#1.33#1.5#4.0/2
  for i, weight in enumerate(weight_list[:-5]):
    print("Weight info ", i, weight.shape, new_weight_list[i].shape)
    if(len(weight.shape)==4):
      #new_weight_list[i] = scale*weight[:,:,-new_weight_list[i].shape[2]:,-new_weight_list[i].shape[3]:]
      new_weight_list[i] = scale*weight[:,:,:new_weight_list[i].shape[2],:new_weight_list[i].shape[3]]

    if(len(weight.shape)==1):
      # new_weight_list[i] = scale*weight[-new_weight_list[i].shape[0]:]
      new_weight_list[i] = scale*weight[:new_weight_list[i].shape[0]]
    if(len(weight.shape)==2):
      # new_weight_list[i] = scale*weight[-new_weight_list[i].shape[0]:, -new_weight_list[i].shape[1]:]
      new_weight_list[i] = scale*weight[:new_weight_list[i].shape[0], :new_weight_list[i].shape[1]]
  new_model.set_weights(new_weight_list)
  for layer in new_model.layers[:-5]: # Reduce this to -1 KJN Change
    layer.trainable = False

  for i, weight in enumerate(weight_list):
    print("Weight value ",i, weight.shape)
    if(len(weight.shape) == 4):
      weight[:,:,:,:int(mask*weight.shape[3])] = 0
      if(i != 0): # KJN Uncomment this
        weight[:,:,:,int(mask*weight.shape[3]):] *= (1/mask)
  model.set_weights(weight_list)
  evaluation = model.evaluate(x_test, y_test)
  print("Evaluation Test Set After ", evaluation)

  for layer in model.layers[:-5]: # Reduce this to -1 KJN Change
    layer.trainable = False

  model = new_model
  # with tf.device('/gpu:0'):
  #   model = gvc.get_conv_net_vert(x_train.shape[1:], num_classes, mask)

  opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
  # Let's train the model using RMSprop
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
  model.summary()

  # with tf.device('/cpu:0'):


  # with tf.device('/gpu:0'):
  #   # model = gwrn.create_wide_residual_network(x_train.shape[1:], 
  #   #   2-resize_factor,
  #   #   nb_classes=num_classes, 
  #   #   wgt_fname="./saved_models/keras_cifar10_weight_0.h5",
  #   #   N=4, k=8, dropout=0.0)

  #   # model = gvc.get_conv_net(x_train.shape[1:], \
  #   #   num_classes, 2-resize_factor,
  #   #   wgt_fname="../../data/conv/saved_model_vgg_v0/keras_cifar10_weight_0.h5")
    
  #   # model = gvc.get_conv_net_v1(x_train.shape[1:], \
  #   #   num_classes, 2-resize_factor,
  #   #   wgt_fname="../../data/conv/saved_model_vgg_v0/keras_cifar10_weight_0.h5")

  #   # model = gvc.get_conv_net_cifar100(x_train.shape[1:], \
  #   #   num_classes, 3-resize_factor,
  #   #   wgt_fname="../../data/conv/cifar100_vgg_v2/cifar100vgg_v1.h5")

  #   # model = gcn.get_conv_net(x_train.shape[1:], \
  #   #   num_classes, 2-resize_factor, 
  #   #   wgt_fname="../../data/conv/mnist_conv_v0/keras_mnist_weight_0.h5")


  #   model = gcn.get_conv_net(x_train.shape[1:], \
  #     num_classes, 2-resize_factor,
  #     wgt_fname=weight_path)

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
        "keras_cifar10_weight_mask_"+str(mask)+".h5")
      # model.save(weight_path)
      
      modelCheckpoint = ModelCheckpoint(weight_path, 
        monitor='val_acc', verbose=0, save_best_only=True, 
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
      #   if epoch%10==0 and epoch>0:
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

