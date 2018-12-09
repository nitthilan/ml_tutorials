from keras.layers import Input, Dense
from keras.models import Model

from keras.datasets import mnist
import numpy as np

from keras.callbacks import ModelCheckpoint


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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_load.ld_data_load as lddl

def get_model(input_dim, encoding_dim):
	# this is our input placeholder
	input_img = Input(shape=(input_dim,))
	enc1 = Dense(75, activation='relu')(input_img)
	# "encoded" is the encoded representation of the input
	encoded = Dense(encoding_dim, activation='relu')(enc1)
	dec1 = Dense(75, activation='relu')(encoded)
	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(input_dim, activation='sigmoid')(dec1)

	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)

	# this model maps an input to its encoded representation
	encoder = Model(input_img, encoded)


	# create a placeholder for an encoded (32-dimensional) input
	encoded_input = Input(shape=(encoding_dim,))
	# retrieve the last layer of the autoencoder model
	decoder_layer = autoencoder.layers[-2]
	# create the decoder model
	decoder = Model(encoded_input, decoder_layer(encoded_input))

	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	return autoencoder, encoder, decoder



def checkpoint(save_dir, encoding_dim):
	weight_path = os.path.join(save_dir, \
		"ld_autoencoder_"+str(encoding_dim)+".h5")
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
	return callbacks

def train_encoder(data, save_dir, x_all, split):
	N_TR = int(split_ld_data.shape[0]*split)
	print("Num train ", N_TR)
	x_train = x_all[:N_TR]
	x_test = x_all[N_TR:]
	
	input_dim = data.shape[1]
	for encoding_dim in [20,30,40,50]: #[2,3,4,5,6,7,8,9,10]:
		print("AutoEncoding ", encoding_dim)

		with tf.device('/gpu:1'):
			autoencoder, encoder, decoder = \
				get_model(input_dim, encoding_dim)


			autoencoder.fit(x_train, x_train,
			                epochs=50 ,
			                batch_size=256,
			                shuffle=True,
			                callbacks=checkpoint(save_dir, encoding_dim),
			                validation_data=(x_test, x_test))

		# encode and decode some digits
		# note that we take them from the *test* set
		encoded_imgs = encoder.predict(x_all)
		# decoded_imgs = decoder.predict(encoded_imgs)
		print("Min and Max", np.min(encoded_imgs, axis=0), \
			np.max(encoded_imgs, axis=0))



save_dir = "../../data/ml_tutorials/autoencoder"

ld_data = lddl.get_data(False, 100000)
# print(ld_data.shape)
split_ld_data = lddl.split_vector(ld_data)
# print(split_ld_data.shape)


train_encoder(split_ld_data, save_dir, split_ld_data, 0.9)