from keras.applications import resnet50
from keras.applications import mobilenetv2
# from keras.applications import mobilenet
from keras.applications import vgg19

# from keras_squeezenet import SqueezeNet
import SqueezeNet as sqn
import get_vgg16_cifar10 as gvc
import gen_conv_net as gcn
import MobileNet as mobilenet
import MobileNet_for_mobile as mobilenet_for_mobile


from keras_applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.engine.input_layer import Input

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.models import Model
from keras import optimizers
import keras

import numpy as np

from os import listdir
from os.path import isfile, join
import os

import matplotlib.image as mpimg
import time

# SqueezeNet: https://github.com/rcmalli/keras-squeezenet/blob/master/examples/example_keras_squeezenet.ipynb
# https://keras.io/applications/

def get_all_nets(network_name, include_top=True):
	if(network_name=="ResNet50"):
		model = resnet50.ResNet50(weights='imagenet',
			include_top=include_top, input_shape=(224, 224, 3))
		# if(include_top==False):
		# 	model.pop()
	elif(network_name=="MobileNetV2"):
		model = mobilenetv2.MobileNetV2(weights='imagenet',
			include_top=include_top, input_shape=(224, 224, 3))
	elif(network_name=="MobileNet"):
		model = mobilenet.MobileNet(weights='imagenet',
			include_top=include_top, input_shape=(224, 224, 3))
	elif(network_name=="VGG19"):
		model = vgg19.VGG19(weights='imagenet',
			include_top=include_top)
	elif(network_name=="SqueezeNet"):
		model = SqueezeNet(weights='imagenet',
		include_top=include_top)
		# if(include_top==False):
		# 	model.pop()
		# 	model.pop()
		# 	model.pop()
		# 	model.pop()

	if(include_top):
		opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

		# Let's train the model using RMSprop
		model.compile(loss='categorical_crossentropy',
		      optimizer=opt,
		      metrics=['accuracy'])
	return model

def get_nets_wo_weights(network_name, num_classes, include_top=False,
	input_shape=(32, 32, 3), num_filter=1, use_bias=False):
	if(network_name=="ResNet50"):
		model = resnet50.ResNet50(include_top=include_top, 
			input_shape=input_shape, weights=None)
		# if(include_top==False):
		# 	model.pop()
	elif(network_name=="MobileNetV2"):
		model = mobilenetv2.MobileNetV2(include_top=include_top, 
			input_shape=input_shape, weights=None,
			classes=num_classes)
	elif(network_name=="MobileNet"):
		model = mobilenet.MobileNet(
			include_top=include_top, input_shape=input_shape, weights=None,
			classes=num_classes, num_filter=num_filter)
	elif(network_name=="MobileNet_for_mobile"):
		model = mobilenet_for_mobile.MobileNet(
			include_top=include_top, input_shape=input_shape, weights=None,
			classes=num_classes, num_filter=num_filter)
	elif(network_name=="VGG19"):
		model = vgg19.VGG19(input_shape=input_shape,
			include_top=include_top, weights=None,
			classes=num_classes)
	elif(network_name=="SqueezeNet"):
		model = sqn.SqueezeNet(input_shape=input_shape,
			include_top=include_top, weights=None, num_filter=num_filter, 
			use_bias=use_bias, classes=num_classes)
	elif(network_name=="vgg"):
		model = gvc.get_conv_vert_net(x_shape=input_shape, 
			num_classes=num_classes, num_vert_filters=num_filter,
    		use_bias=use_bias)
	elif(network_name=="conv"):
		model = gcn.get_conv_vert_net(input_shape=input_shape, 
			num_classes=num_classes, 
			num_extra_conv_layers=2, num_ver_filter=num_filter, 
			use_bias=use_bias)

	if(include_top == False):
		x = model.output
		# x = keras.layers.GlobalAveragePooling2D()(x)
		x = Flatten()(x)
		x = Dense(256, activation='relu')(x)
		# x = Activation('relu')(x)
		x = Dropout(0.5)(x)
		# x = Dense(num_output)(x)
		# x = Activation('softmax')(x)
		x = keras.layers.Dense(num_classes, activation='softmax',
	                         use_bias=True, name='Logits')(x)
	
		full_model = Model(inputs = model.input,outputs = x)
	else:
		full_model = model
	opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	full_model.compile(loss='categorical_crossentropy',
	      optimizer=opt,
	      metrics=['accuracy'])
	return full_model



def preprocess_image(network_name, x):
	if(network_name=="ResNet50"):
		x = resnet50.preprocess_input(x)
	elif(network_name=="MobileNetV2"):
		x = mobilenetv2.preprocess_input(x)
	elif(network_name=="MobileNet"):
		x = mobilenet.preprocess_input(x)
	elif(network_name=="VGG19"):
		x = vgg19.preprocess_input(x)
	elif(network_name=="SqueezeNet"):
		x = imagenet_utils.preprocess_input(x)
	return x

def decodepred(network_name, preds):
	if(network_name=="ResNet50"):
		preds = resnet50.decode_predictions(preds, top=3)[0]
	elif(network_name=="MobileNetV2"):
		preds = mobilenetv2.decode_predictions(preds, top=3)[0]
	elif(network_name=="MobileNet"):
		preds = mobilenet.decode_predictions(preds, top=3)[0]
	elif(network_name=="VGG19"):
		preds = vgg19.decode_predictions(preds, top=3)[0]
	elif(network_name=="SqueezeNet"):
		preds = imagenet_utils.decode_predictions(preds, top=3)[0]
	return x

def analyse_model(model):
	print("All functions ", dir(model))
	print("Summary model ", model.summary())
	print("Layer details ", dir(model.layers[2]))
	for i, layer in enumerate(model.layers):
		print("Length in each layer ", i, layer.name,
			layer.input_shape, layer.output_shape,
			len(layer.weights))
		if(len(layer.weights)):
			for j, weight in enumerate(layer.weights):
				print("Weights ", j, weight.shape)
	return

def add_classifier(base_model, num_output):

	for layer in base_model.layers:
		layer.trainable = False
		
	x = base_model.output
	x = keras.layers.GlobalAveragePooling2D()(x)
	# x = Dense(16, kernel_regularizer=regularizers.l2(0.01))(x)
	# x = Activation('relu')(x)
	# x = Dropout(0.5)(x)
	# x = Dense(num_output)(x)
	# x = Activation('softmax')(x)
	x = keras.layers.Dense(num_output, activation='softmax',
                         use_bias=True, name='Logits')(x)
	
	model = Model(inputs = base_model.input,outputs = x)
	# initiate RMSprop optimizer
	opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='binary_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])
	return model

def get_all_prediction(image_filelist):
	prediction_list = []
	for filename in image_filelist:

		# img = image.load_img(os.path.join(imagenet_path, filename), target_size=(224, 224))
		img = image.load_img(os.path.join(imagenet_path, filename), target_size=(227, 227)) # Squeezenet
		# img1 = mpimg.imread(os.path.join(imagenet_path, filename))
		# print(img1.shape)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = imagenet_utils.preprocess_input(x)

		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		print('Predicted:', filename, imagenet_utils.decode_predictions(preds, top=3)[0])
		print("Pred values ", np.argmax(preds))
		# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
		prediction_list.append(preds)
	return prediction_list


if __name__ == '__main__':

	network_types_list = ["MobileNetV2"]#, "ResNet50", "MobileNetV2", "VGG19"] # , "SqueezeNet"
	for network_type in network_types_list:
		print("Network Type ", network_type)
		model = get_all_nets(network_type, include_top=True)
		analyse_model(model)
		# model = get_all_nets(network_type, include_top=False)
		# model = add_classifier(model)

	imagenet_path = "/mnt/additional/aryan/imagenet_validation_data/ILSVRC2012_img_val/"
	# http://www.image-net.org/challenges/LSVRC/2012/
	# https://cv-tricks.com/tensorflow-tutorial/keras/
	# Finding actual predictions
	# http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html

	image_filelist = [f for f in listdir(imagenet_path) if isfile(join(imagenet_path, f))]

	print("Number of files ", len(image_filelist))


	start_time = time.time()
	get_all_prediction(image_filelist[:10])
	total_time = time.time() - start_time
	print("Total prediction time ", total_time)

	print("File list ", image_filelist[:10])