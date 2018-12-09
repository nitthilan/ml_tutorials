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

# import SqueezeNet as sqn


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


network = sys.argv[1] #"MobileNetV2", "MobileNet" "SqueezeNet" "VGG19" "vgg" "conv"
data_set = sys.argv[2] #mnist, cifar10, cifar100
num_filter = float(sys.argv[3])

use_bias = False # sys.argv[1].lower() == 'true' # 
batch_size = 128 #32
# num_classes = 100 # 10 for cifar10 # 100 for cifar100
epochs = 1
data_augmentation = True
# num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_keras_models')
model_name = network+'_'+data_set+'_false_20112018_chk'

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

x_train, y_train, x_test, y_test = \
	gkd.get_data(data_set)
if(network == "SqueezeNet"):
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

# x_train /= 256
# x_test /= 256

print("Input info ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# with tf.device('/cpu:0'):
with tf.device('/gpu:0'):
	model = gai.get_nets_wo_weights(network, num_classes, input_shape=input_shape, 
		num_filter=num_filter, include_top=True)
	weight_list = model.get_weights()
	for i, weight in enumerate(weight_list):
			print("Weight info ", i, weight.shape)
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
		monitor='val_acc', verbose=0, save_best_only=True, 
		save_weights_only=False, mode='auto', period=1)

	callbacks = [
	          modelCheckpoint
	          #   earlyStopping, 
	          #   reduceonplateau,
	          #   csv_logger
	          ]

	# initiate RMSprop optimizer
	# optimizers = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
	# optimizers = keras.optimizers.Adam(lr=0.01, decay=1e-6)
	optimizers = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	parallel_model.compile(loss='categorical_crossentropy', \
	optimizer=optimizers, metrics=['accuracy'])

	parallel_model.fit_generator(datagen.flow(x_train, y_train,
	                               batch_size=batch_size),
	                  steps_per_epoch=x_train.shape[0] // batch_size,
	                  epochs=epochs,
	                  validation_data=(x_test, y_test),
	                  callbacks = callbacks)
	print('Saved trained model and weights at %s ' % weight_path)

	output_tflite_model = weight_path[:-3]+".tflite"
	converter = tf.contrib.lite.TocoConverter.from_keras_model_file(weight_path)
	tflite_model = converter.convert()
	open(output_tflite_model, "wb").write(tflite_model)