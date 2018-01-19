
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from PIL import Image


import os
import pickle
import numpy as np


def resize_images(x_train, resize_factor):
	(n,w,h,d) = x_train.shape
	w = int(w*1.0/resize_factor)
	h = int(h*1.0/resize_factor)
	# print("Image dimensions", n,w,h,d)
	x_train_resized = np.zeros((n,w,h,d))
	for idx in range(n):
		im = Image.fromarray(np.uint8(x_train[idx,:,:,:]))
		x_train_resized[idx,:,:,:] = im.resize((w,h), Image.ANTIALIAS)
	return x_train_resized

def get_cifar10_data(resize_factor):
	# The data, shuffled and split between train and test sets:
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	num_classes = int(np.max(y_train)+1)

	if(resize_factor):
		x_train = resize_images(x_train, 2**resize_factor)
		x_test = resize_images(x_test, 2**resize_factor)
		# im.save("resize_"+str(idx)+".jpg")

	# print('x_train shape:', x_train.shape, num_classes)
	# print(x_train.shape[0], 'train samples')
	# print(x_test.shape[0], 'test samples')

	# Convert class vectors to binary class matrices.
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)


	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	return x_train, y_train, x_test, y_test

if __name__ == '__main__':
	get_cifar10_data(2)