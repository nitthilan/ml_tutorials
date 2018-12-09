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

from keras.models import load_model

import SqueezeNet as sqn


from keras import backend as K
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


import gen_conv_net as gcn
# import get_data as gd
import get_vgg16_cifar10 as gvc
import get_wide_res_networks as gwrn
import get_lenet as gln
import get_all_imagenet as gai


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.get_keras_data as gkd

# Batch Normalisation
# https://www.youtube.com/watch?v=gYpoJMlgyXA&t=3078
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
# https://github.com/keras-team/keras/issues/1523

# CIFAR10 performance
# https://stackoverflow.com/questions/52209851/expected-validation-accuracy-for-keras-mobile-net-v1-for-cifar-10-training-from

#

def get_mask_weights(weight_list_copy, mask):
	for i,weight in enumerate(weight_list_copy):
		if(len(weight.shape) == 4 and i != 0):
			if(weight.shape[0] == 3 and weight.shape[1] == 3):
				weight[:,:,int(mask*weight.shape[2]/4):,:] = 0
				# weight[:,:,:int(mask*weight.shape[2]/4),:] *= (4/mask)
				weight_list_copy[i+1][int(mask*weight.shape[2]/4):] = 0
				weight_list_copy[i+2][int(mask*weight.shape[2]/4):] = 0
				weight_list_copy[i+3][int(mask*weight.shape[2]/4):] = 0
				weight_list_copy[i+4][int(mask*weight.shape[2]/4):] = 0
				print("Layer modified 3x3 ", i, int(mask*weight.shape[2]/4), 4/mask)

			if(weight.shape[0] == 1 and weight.shape[1] == 1):
				weight[:,:,:,int(mask*weight.shape[3]/4):] = 0
				weight[:,:,:,:int(mask*weight.shape[3]/4)] *= (4/mask)
				weight_list_copy[i+1][int(mask*weight.shape[3]/4):] = 0
				weight_list_copy[i+2][int(mask*weight.shape[3]/4):] = 0
				weight_list_copy[i+3][int(mask*weight.shape[3]/4):] = 0
				weight_list_copy[i+4][int(mask*weight.shape[3]/4):] = 0
				print("Layer modified 1x1 ", i, int(mask*weight.shape[3]/4), 4/mask)

			# if(weight.shape[0] == 3 and weight.shape[1] == 3):
			# 	for j in range(weight.shape[2]-mask, weight.shape[2]):
			# 		weight[:,:,j,:] = 0
			# if(weight.shape[0] == 1 and weight.shape[1] == 1):
			# 	for j in range(weight.shape[3]-mask, weight.shape[3]):
			# 		weight[:,:,:,j] = 0
		print("Weight shape ", i, weight.shape)
	return weight_list_copy

def get_mask_vgg(weight_list_copy, mask):
	
	for i in range(4,31):
		weight_shape = weight_list_copy[i].shape
		if(len(weight_shape) == 4):
			weight_list_copy[i][:,:,:,int(mask*weight_shape[3]/4):] = 0
			weight_list_copy[i][:,:,:,:int(mask*weight_shape[3]/4)] *= (4/mask)
			print("Mask ", i, int(mask*weight_shape[3]/4))
		# weight_shape = weight_list_copy[28].shape
		# weight_list_copy[28][int(mask*weight_shape[3]/4):] = 0
		# weight_shape = weight_list_copy[28].shape
		# weight_list_copy[28][int(mask*weight_shape[3]/4):] = 0
	for i,weight in enumerate(weight_list_copy):
		print("Weight shape ", i, weight.shape)
	# for i,weight in enumerate(weight_list_copy):
	# 	if(len(weight.shape) == 4):
	# 		weight[:,:,:,int(mask*weight.shape[3]/4):] = 0
	# 		# weight_list_copy[i+1][int(mask*weight.shape[3]/4):] = 0
	# 		weight[:,:,:,:int(mask*weight.shape[3]/4)] *= (4/mask)
	# 		if(i != 0):
	# 			filt_offset = int(mask*weight_list_copy[i-2].shape[3]/4)
	# 			weight[:,:,filt_offset:,:] = 0
	# 			print("Filt Offset ", filt_offset)
	# 		# weight_list_copy[i+1][int(mask*weight.shape[3]/4):] = 0
	# 		print("Layers modified ", int(mask*weight.shape[3]/4), (4/mask))
	# 	# if(len(weight.shape) == 4 and i != 0):
	# 	# 	if(weight.shape[0] == 3 and weight.shape[1] == 3):
	# 	# 		for j in range(int(mask*weight.shape[2]/4), weight.shape[2]):
	# 	# 			weight[:,:,j,:] = 0
	# 	# 			weight_list_copy[i+1][j] = 0
	# 	# 			weight_list_copy[i+2][j] = 0
	# 	# 			weight_list_copy[i+3][j] = 0
	# 	# 			weight_list_copy[i+4][j] = 0
	# 	# 	if(weight.shape[0] == 1 and weight.shape[1] == 1):
	# 	# 		for j in range(int(mask*weight.shape[3]/4), weight.shape[3]):
	# 	# 			weight[:,:,:,j] = 0
	# 	# 			weight_list_copy[i+1][j] = 0
	# 	# 			weight_list_copy[i+2][j] = 0
	# 	# 			weight_list_copy[i+3][j] = 0
	# 	# 			weight_list_copy[i+4][j] = 0
	# 	# 	print("Layer modified ", i, int(mask*weight.shape[2]/4), weight.shape[2],
	# 	# 		(mask*weight.shape[3]/4), weight.shape[3])

	# 	print("Weight shape ", i, weight.shape)
	return weight_list_copy






network = sys.argv[1] #"MobileNetV2", "MobileNet" "SqueezeNet" "VGG19"
data_set = sys.argv[2] #mnist, cifar10, cifar100
num_filter = int(sys.argv[3])
use_bias = False

batch_size = 128 #32
# num_classes = 100 # 10 for cifar10 # 100 for cifar100
epochs = 200
data_augmentation = True
# num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_keras_models')
model_name = network+'_'+data_set+'_false_'

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

x_train, y_train, x_test, y_test = \
	gkd.get_data(data_set)
x_train_48 = np.zeros((x_train.shape[0], 48, 48, x_train.shape[3]))
x_test_48 = np.zeros((x_test.shape[0], 48, 48, x_test.shape[3]))
x_train_48[:, 8:40, 8:40, :] = x_train
x_test_48[:, 8:40, 8:40, :] = x_test
x_train = x_train_48
x_test = x_test_48

num_classes = int(y_train.shape[1])
input_shape = x_train.shape[1:]
# x_train = gai.preprocess_image(network, x_train)
# x_test = gai.preprocess_image(network, x_test)


x_train /= 128
x_train -= 1
x_test /= 128
x_test -= 1

print("Input info ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# with tf.device('/cpu:0'):
with tf.device('/gpu:0'):
	if(num_filter == 1 or num_filter == 2 or num_filter == 3):
		# model = gai.get_nets_wo_weights(network, num_classes, 
		# 	input_shape=input_shape, num_filter=num_filter)
		model = sqn.SqueezeNet(input_shape=input_shape,
			include_top=True, weights=None, num_filter=num_filter, 
			use_bias=False, classes=num_classes)
		weight_list = model.get_weights()
		num_weights_to_ignore = 0
		for i, weight in enumerate(weight_list):
  			print("Weight info ", i, weight.shape)
	else:
		weight_path_1 = os.path.join(save_dir, model_name+str(1)+".h5")
		model = load_model(weight_path_1)
		score = model.evaluate(x_test , y_test, 
			batch_size=batch_size, verbose=1)
		print("Score ", model_name, score)
		model.summary()
		weight_list = model.get_weights() 
		weight_list_copy = np.copy(weight_list)#copy.deepcopy(weight_list)
		# weight_list_copy = get_mask_weights(weight_list_copy, 5-num_filter)
		weight_list_copy = get_mask_vgg(weight_list_copy, 5-num_filter)
		model.set_weights(weight_list_copy)
		for layer in model.layers[:-3]:
			layer.trainable = False
		opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

		# Let's train the model using RMSprop
		model.compile(loss='categorical_crossentropy',
		      optimizer=opt,
		      metrics=['accuracy'])
	model.summary()

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
	model_name+str(num_filter)+".h5")
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
	print('Saved trained model and weights at %s ' % weight_path)


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

