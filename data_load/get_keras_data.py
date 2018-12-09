
from __future__ import print_function
import keras
from keras.datasets import cifar10, cifar100, mnist
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


def get_mean_std(dataname):
	if(dataname == "mnist"):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
	elif(dataname == "cifar10"):
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	else:
		(x_train, y_train), (x_test, y_test) = cifar100.load_data()
	mean = np.mean(x_train)
	std = np.std(x_train)
	return mean, std

def get_data(dataname):
	if(dataname == "mnist"):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
	elif(dataname == "cifar10"):
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	else:
		(x_train, y_train), (x_test, y_test) = cifar100.load_data()
	num_classes = int(np.max(y_train)+1)

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	# x_train = (x_train - mean)/std
	# x_test = (x_test - mean)/std
	if(dataname=="mnist"):
		x_train_zeros = np.zeros((60000, 32, 32, 1))
		x_test_zeros = np.zeros((10000, 32, 32, 1))

		x_train_zeros[:,2:30, 2:30, 0] = x_train
		x_test_zeros[:,2:30, 2:30, 0] = x_test
		x_train = x_train_zeros
		x_test = x_test_zeros

	return x_train, y_train, x_test, y_test

def get_cifar_data(resize_factor, num_classes):
	# The data, shuffled and split between train and test sets:
	if(num_classes == 10):
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	else:
		(x_train, y_train), (x_test, y_test) = cifar100.load_data()
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

def get_reduced_class_data(total_class_types, \
	selec_class_list):
	if(total_class_types == 10):
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	else:
		(x_train, y_train), (x_test, y_test) = cifar100.load_data()

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	tr_te_list = []
	for x, y in zip([x_train, x_test],[y_train, y_test]):
		y = np.squeeze(y)
		conc_list_x = []
		conc_list_y = []
		for idx, elem in enumerate(selec_class_list):
			conc_list_x.append(x[np.where(y==elem)])
			conc_list_y.append(idx*np.ones(conc_list_x[-1].shape[0]))
		x = np.concatenate(conc_list_x,axis=0)
		y = np.concatenate(conc_list_y,axis=0)
		y = np.expand_dims(y, axis=1)
		y = keras.utils.to_categorical(y, len(selec_class_list))
		print(x.shape, y.shape)
		print(y[:10], y[-10:])
		tr_te_list.append((x,y))

	return tr_te_list[0], tr_te_list[1]

def scale_image(x_train, x_test):
	mean = np.mean(x_train)
	std = np.std(x_train)
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
	# dump_images(1, "../../data/conv/resize_16_0312")
	print(get_mean_std("cifar10"))
	print (get_mean_std("cifar100"))
	print (get_mean_std("mnist"))