from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import tensorflow as tf
from keras.utils import multi_gpu_model

from keras import backend as K
import get_all_imagenet as gai
import pickle
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.get_keras_data as gkd

import datetime
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.models import load_model



load_cifar10_list = [
	"./saved_keras_models/conv_cifar10_false_4.h5",
	"./saved_models_bkup/vert_filt_conv_cifar10_2.h5",
	"./saved_models_bkup/vert_filt_conv_cifar10_3.h5",
	"./saved_keras_models/vgg_cifar10_false_4.h5",
	"./saved_models_bkup/vert_filt_vgg16_cifar10_2.h5",
	"./saved_models_bkup/vert_filt_vgg16_cifar10_3.h5",
	# "./saved_models_bkup/vert_filt_vgg16_cifar10_4.h5",
]
load_mnist_list = [
	"./saved_keras_models/conv_mnist_false_4.h5",
	"./saved_models_bkup/vert_filt_conv_mnist_2.h5",
	"./saved_models_bkup/vert_filt_conv_mnist_3.h5",
]
load_cifar100_list = [
	"./saved_keras_models/conv_cifar100_false_4.h5",
]
load_squeeze_cifar10_list = [
	"./saved_keras_models/SqueezeNet_cifar10_false_1.h5",
	"./saved_models_bkup/vert_filt_squeezenet_cifar10_2.h5",
	"./saved_models_bkup/vert_filt_squeezenet_cifar10_3.h5",
]
load_squeeze_cifar10_list = [
	"./saved_keras_models/SqueezeNet_cifar10_false_20112018_1.h5",
	"./saved_models/vert_filt_SqueezeNet_cifar10_3.0.h5",
	"./saved_models/vert_filt_SqueezeNet_cifar10_2.0.h5",
]
load_squeeze_mnist_list = [
	"./saved_keras_models/SqueezeNet_mnist_false_1.h5",
	"./saved_models_bkup/vert_filt_squeezenet_mnist_2_12_1.h5",
	"./saved_models_bkup/vert_filt_squeezenet_mnist_3.h5",
]
load_mobile_cifar100_list = [
	"./saved_keras_models/MobileNet_cifar100_false_8.h5"
]
load_mobile_cifar10_list = [
	"./saved_keras_models/MobileNet_cifar10_false_4.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_2.0.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_2.5.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_3.0.h5",
	"./saved_models/vert_filt_MobileNet_cifar10_3.5.h5",
]
load_mobile_mnist_list = [
	"./saved_keras_models/MobileNet_mnist_false_4.h5",
	"./saved_models_bkup/vert_filt_MobileNet_mnist_2.0.h5",
	"./saved_models_bkup/vert_filt_MobileNet_mnist_2.5.h5",
	# "./saved_models/vert_filt_MobileNet_mnist_3.0.h5",
	"./saved_models_bkup/vert_filt_MobileNet_mnist_3.5.h5",
	"./saved_models_bkup/vert_filt_MobileNet_mnist_2.75.h5",
	"./saved_models_bkup/vert_filt_MobileNet_mnist_1.0.h5"
]
load_mobile_mnist_list = [
	"./saved_keras_models/MobileNet_mnist_false_20112018_4.h5", 
	"./saved_models/vert_filt_MobileNet_mnist_2.0.h5",
	"./saved_models/vert_filt_MobileNet_mnist_2.5.h5",
	"./saved_models/vert_filt_MobileNet_mnist_3.0.h5",
	"./saved_models/vert_filt_MobileNet_mnist_3.5.h5",
]

def get_dataset(dataset, model_name):
	x_train, y_train, x_test, y_test = gkd.get_data(dataset)

	if(model_name == "SqueezeNet"):
		x_train_48 = np.zeros((x_train.shape[0], 48, 48, x_train.shape[3]))
		x_test_48 = np.zeros((x_test.shape[0], 48, 48, x_test.shape[3]))
		x_train_48[:, 8:40, 8:40, :] = x_train
		x_test_48[:, 8:40, 8:40, :] = x_test
		x_train = x_train_48
		x_test = x_test_48
	# Preprocessing of data
	x_train /= 128
	x_train -= 1
	x_test /= 128
	x_test -= 1

	# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
	return x_train, y_train, x_test, y_test


def evaluate_models(load_weight_path_list, x_test, y_test):
	for load_weight_path in load_weight_path_list:
		with tf.device('/gpu:0'):
			trained_model = load_model(load_weight_path)
			weight_list = trained_model.get_weights()

		print(load_weight_path)

		print("Start ", datetime.datetime.now())
		evaluation = trained_model.evaluate(x_test, y_test, batch_size=128, verbose=1)
		print("End ",datetime.datetime.now())
		print("Evaluation Test Set ", evaluation)
		# time.sleep(10)

		# trained_model.summary()
	return

def predict_models(load_weight_path_list, x_test, y_test):
	for load_weight_path in load_weight_path_list:
		with tf.device('/gpu:0'):
			trained_model = load_model(load_weight_path)
			weight_list = trained_model.get_weights()

		print(load_weight_path)

		print("Start ", datetime.datetime.now())
		prediction = trained_model.predict(x_test, batch_size=128, verbose=1)
		print("End ",datetime.datetime.now())
		predict_path = load_weight_path[:-3]+"_predict.npz"
		np.savez(predict_path, prediction, y_test)
		print(predict_path)

		# print("Evaluation Test Set ", evaluation)
		# trained_model.summary()
	return

load_list_list =[
	load_cifar10_list, load_mnist_list, 
	load_squeeze_cifar10_list, load_squeeze_mnist_list,
	load_mobile_cifar100_list, load_mobile_cifar10_list,
	load_mobile_mnist_list
]
model_name_list = ["conv_vgg", "conv_vgg", "SqueezeNet", "SqueezeNet", 
	"MobileNet", "MobileNet", "MobileNet"]
dataset_list = ["cifar10", "mnist", "cifar10", "mnist", "cifar100", 
	"cifar10", "mnist"]
# nvidia-smi --query-gpu=power.draw,utilization.gpu,timestamp --format=csv --loop-ms=10 -i 1
for load_list, model_name, dataset in \
	zip(load_list_list[4:], model_name_list[4:], dataset_list[4:]):
	# zip(load_list_list[2:4], model_name_list[2:4], dataset_list[2:4]):
	# zip(load_list_list[:4], model_name_list[:4], dataset_list[:4]):

	x_train, y_train, x_test, y_test = \
		get_dataset(dataset, model_name)
	evaluate_models(load_list, x_test, y_test)
	# evaluate_models(load_list, x_train, y_train)
	# predict_models(load_list, x_test, y_test)
	# predict_models(load_list, x_train, y_train)

	# To get energy numbers
	# for i in range(5):
	# 	evaluate_models(load_list, x_train, y_train)

print(datetime.datetime.now())