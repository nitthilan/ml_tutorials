'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
# from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import tensorflow as tf
from keras.utils import multi_gpu_model
from numpy import linalg as LA


from keras import backend as K
# import get_wide_res_networks as gwrn
from keras.models import load_model, save_model




import pickle
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import data_load.get_keras_data as gkd
# import get_vgg16_cifar10 as gvc
# import gen_conv_net as gcn
import conv.networks.get_all_imagenet as gai

# for num_filter in [0.5, 0.625, 0.75, 0.875, 1.0]:
# 	base_path = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/"
# 	file_name = "MobileNet"+str(num_filter)+".h5"
# 	trained_model = gai.get_all_nets("MobileNet", include_top=True)
# 	weight_list = trained_model.get_weights()

# 	dst_model = gai.get_all_nets("MobileNet_for_mobile", include_top=True, num_filter=4*num_filter)
# 	dst_weight_list = dst_model.get_weights()

# 	for i,weight in enumerate(weight_list):
# 		dst_shape = dst_weight_list[i].shape
# 		if(len(weight.shape) == 4):
# 			dst_weight_list[i] = weight[:,:,:dst_shape[2],:dst_shape[3]]
# 		else:
# 			dst_weight_list[i] = weight[:dst_shape[0]]
# 		print(i, weight.shape, dst_weight_list[i].shape)
# 		# dst_weight_list[i] = weight_list[i]

# 	dst_model.set_weights(dst_weight_list)
# 	dst_file_name = "MobileNet_for_mobile"+str(num_filter)+".h5"
# 	dst_model.save(base_path+dst_file_name)
# 	print(base_path, file_name, dst_file_name)


for num_filter in [0.5, 0.625, 0.75, 0.875, 1.0]:
# num_filter = 1.0
	base_path = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/"
	# file_name = "MobileNet"+str(num_filter)+".h5"
	trained_model = gai.get_all_nets("VGG19", include_top=True)
	weight_list = trained_model.get_weights()

	dst_model = gai.get_all_nets("VGG19_for_mobile", include_top=True, num_filter=num_filter)
	dst_weight_list = dst_model.get_weights()

	for i,weight in enumerate(weight_list):
		dst_shape = dst_weight_list[i].shape
		if(len(weight.shape) == 4):
			dst_weight_list[i] = weight[:,:,:dst_shape[2],:dst_shape[3]]
		elif(len(weight.shape) == 2):
			dst_weight_list[i] = weight[:dst_shape[0],:dst_shape[1]]
		else:
			dst_weight_list[i] = weight[:dst_shape[0]]
		print(i, weight.shape, dst_shape, dst_weight_list[i].shape)
		# dst_weight_list[i] = weight_list[i]

	dst_model.set_weights(dst_weight_list)
	dst_file_name = "VGG19_for_mobile"+str(num_filter)+".h5"
	dst_model.save(base_path+dst_file_name)
	print(base_path, dst_file_name)

