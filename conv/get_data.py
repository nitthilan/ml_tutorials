
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
		im = im.resize((w,h), Image.ANTIALIAS)
		im = np.asarray(im)
		# im = np.roll(im, 2, axis=-1)
		# im = np.transpose(im, [0, 3, 1, 2])
		x_train_resized[idx,:,:,:] = im
	return x_train_resized

def dump_image_folder(x_train, folder_path):
	(n,w,h,d) = x_train.shape
	for idx in range(n):
		im = Image.fromarray(np.uint8(x_train[idx,:,:,:]))
		filename = os.path.join(folder_path, str(idx) + ".bmp")
		im.save(filename)

	return

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

def scale_image(x_train, x_test):
	mean = np.mean(x_train,axis=(0,1,2,3))
	std = np.std(x_train, axis=(0, 1, 2, 3))
	x_train = (x_train-mean)/(std+1e-7)
	x_test = (x_test-mean)/(std+1e-7)

	return x_train, x_test
def dump_images(resize_factor, base_folder):
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	np.savetxt(os.path.join(base_folder,'train_label.txt'), (y_train.astype(int)), fmt='%d')
	np.savetxt(os.path.join(base_folder,'test_label.txt'), (y_test.astype(int)), fmt='%d')

	# print(y_train.shape, y_test.shape)
	if(resize_factor):
		x_train = resize_images(x_train, 2**resize_factor)
		x_test = resize_images(x_test, 2**resize_factor)
	dump_image_folder(x_train, os.path.join(base_folder, "train"))
	dump_image_folder(x_test, os.path.join(base_folder, "test"))

	return

if __name__ == '__main__':
	# get_cifar10_data(2)
	# dump_images(0, "../../data/conv/resize_32/")
	# dump_images(1, "../../data/conv/resize_16/")
	# dump_images(2, "../../data/conv/resize_08/")
	# dump_images(1, "../../data/conv/resize_16_rgb")
	dump_images(1, "../../data/conv/resize_16_0312")