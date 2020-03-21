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

num_classes = 10

for dataset in ["mnist", "svhn", "cifar10"]:
	for num_filter in [2.0, 2.25, 2.5, 3.0, 3.5, 4.0]:
		if(dataset != "cifar10" and num_filter == 2.25):
			continue
		base_path = "./conv/vert_filt_models/"
		file_name = "vert_filt_"+"MobileNet"+"_"+dataset+"_"+str(num_filter)+".h5"
		x_train, y_train, x_test, y_test = gkd.get_data(dataset)
		trained_model = load_model(base_path+file_name)
		weight_list = trained_model.get_weights()

		dst_model = gai.get_nets_wo_weights("MobileNet_for_mobile", num_classes, 
		  input_shape=x_train.shape[1:], num_filter=num_filter, include_top=True)

		dst_model.set_weights(trained_model.get_weights())
		dst_file_name = "vert_filt_"+"MobileNet_for_mobile"+"_"+dataset+"_"+str(num_filter)+".h5"
		dst_model.save(base_path+dst_file_name)
		print(base_path, file_name, dst_file_name)

