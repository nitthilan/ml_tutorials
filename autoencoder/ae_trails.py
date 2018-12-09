
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

from keras.datasets import mnist
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
import random as rd

import sys, os
from keras import backend as K
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

def get_conv_model(input_dim, encoding_dim):
	# Create a 2conv+1relu based stack of auto encoder
	return autoencoder, encoder, decoder

def get_model(input_dim, encoding_dim):
	# this is our input placeholder
	input_img = Input(shape=(input_dim,))
	# "encoded" is the encoded representation of the input
	enc1 = Dense(200, activation='relu')(input_img)
	encoded = Dense(encoding_dim, activation='relu')(enc1)
	# "decoded" is the lossy reconstruction of the input
	dec1 = Dense(200, activation='relu')(encoded)
	decoded = Dense(input_dim, activation='sigmoid')(dec1)

	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)

	# this model maps an input to its encoded representation
	encoder = Model(input_img, encoded)


	# create a placeholder for an encoded (32-dimensional) input
	encoded_input = Input(shape=(encoding_dim,))
	# retrieve the last layer of the autoencoder model
	decoder_layer = autoencoder.layers[-2](encoded_input)
	decoder_layer = autoencoder.layers[-1](decoder_layer)
	# create the decoder model
	decoder = Model(encoded_input, decoder_layer)

	autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

	return autoencoder, encoder, decoder

def get_k_ones(N, D, K):
	pos_sam = np.zeros((N, D))
	# pos_sam = -1*np.ones((N, D))
	for i in range(N):
		pos_sam[i,rd.sample(range(D), K)] = 1

	# mean = np.sum(pos_sam[0])*1.0/D
	# pos_sam = pos_sam - mean
	# noise = np.random.normal(loc=0.0, scale=0.001, size=(N, D))
	# pos_sam += noise
	return pos_sam

N =   1000000
N_TR = 900000
num_ones = 4
input_dim = 30
# this is the size of our encoded representations
encoding_dim = 7  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
num_epochs = 30

num_ones_num_valid_encoding = []
for num_ones in [2]:#[2,3,4,5,6]:
	num_valid_encodings = []


	x_all = get_k_ones(N, input_dim, num_ones)
	x_train = x_all[:N_TR]
	x_test = x_all[N_TR:]
	for encoding_dim in [4]:#[2,3,4,5,6,7,8,9,10]:
		print("AutoEncoding ", num_ones, encoding_dim, x_all.shape)

		with tf.device('/gpu:1'):
			autoencoder, encoder, decoder = \
				get_model(input_dim, encoding_dim)


			autoencoder.fit(x_train, x_train,
			                epochs=num_epochs ,
			                batch_size=256,
			                shuffle=True,
			                validation_data=(x_test, x_test))

		# encode and decode some digits
		# note that we take them from the *test* set
		encoded_imgs = encoder.predict(x_all[:100])
		decoded_imgs = decoder.predict(encoded_imgs)

		decoded_imgs_cpy = decoded_imgs.copy()
		decoded_imgs_cpy[decoded_imgs>=0.5] = 1
		decoded_imgs_cpy[decoded_imgs<0.5]  = 0 

		print(x_all.shape)
		print(encoded_imgs.shape)
		print(decoded_imgs.shape)
		print(x_all[:2])
		print(encoded_imgs[:2])
		print(decoded_imgs[:2])
		print(decoded_imgs_cpy[:2])
		print(np.sum(decoded_imgs_cpy, axis=1))
