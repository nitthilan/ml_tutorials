from keras.layers import Input, Dense
from keras.models import Model

from keras.datasets import mnist
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
import random as rd


def get_model(input_dim, encoding_dim):
	# this is our input placeholder
	input_img = Input(shape=(input_dim,))
	# "encoded" is the encoded representation of the input
	encoded = Dense(encoding_dim, activation='relu')(input_img)
	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(input_dim, activation='sigmoid')(encoded)

	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)

	# this model maps an input to its encoded representation
	encoder = Model(input_img, encoded)


	# create a placeholder for an encoded (32-dimensional) input
	encoded_input = Input(shape=(encoding_dim,))
	# retrieve the last layer of the autoencoder model
	decoder_layer = autoencoder.layers[-1]
	# create the decoder model
	decoder = Model(encoded_input, decoder_layer(encoded_input))

	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	return autoencoder, encoder, decoder

# (x_train, _), (x_test, _) = mnist.load_data()

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print x_train.shape
# print x_test.shape

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
input_dim = 10
# this is the size of our encoded representations
encoding_dim = 7  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

num_ones_num_valid_encoding = []
for num_ones in [2]:#,3,4,5,6]:
	num_valid_encodings = []


	x_all = get_k_ones(N, input_dim, num_ones)
	x_train = x_all[:N_TR]
	x_test = x_all[N_TR:]
	for encoding_dim in [4]:#[2,3,4,5,6,7,8,9,10]:
		print("AutoEncoding ", num_ones, encoding_dim)
		autoencoder, encoder, decoder = \
			get_model(input_dim, encoding_dim)


		autoencoder.fit(x_train, x_train,
		                epochs=50,
		                batch_size=256,
		                shuffle=True,
		                validation_data=(x_test, x_test))

		# encode and decode some digits
		# note that we take them from the *test* set
		encoded_imgs = encoder.predict(x_test)
		decoded_imgs = decoder.predict(encoded_imgs)


		n = 10  # how many digits we will display
		# print(encoded_imgs[:n])
		print(np.min(encoded_imgs, axis=0), \
			np.max(encoded_imgs, axis=0))
		# print(decoded_imgs[:n])
		output = np.copy(decoded_imgs)
		output[decoded_imgs<0.5] = 0
		output[decoded_imgs>0.5] = 1
		# print(output[:n])
		output_sum = np.sum(output, axis=1)
		# print(output_sum[:n])
		num_equal = (output_sum == num_ones)
		# print(num_equal[:n])
		print("validation", np.sum(num_equal))
		num_valid_encodings.append(np.sum(num_equal))

		n_validate = 100000
		min_val = np.min(encoded_imgs, axis=0)
		max_val = np.max(encoded_imgs, axis=0)
		rand_val = np.random.uniform(0,1,size=(n_validate, encoding_dim))
		latent_val = np.multiply(rand_val, max_val)
		print(rand_val[0], latent_val[0], max_val)
		print(latent_val.shape)
		decoded_imgs = decoder.predict(latent_val)
		output = np.copy(decoded_imgs)
		output[decoded_imgs<0.5] = 0
		output[decoded_imgs>0.5] = 1
		# print(output[:n])
		output_sum = np.sum(output, axis=1)
		# print(output_sum[:n])
		num_equal = (output_sum == num_ones)
		print(np.sum(num_equal), n_validate)
		print(output[:10])
	# print(num_ones, num_valid_encodings)
	num_ones_num_valid_encoding.append(num_valid_encodings)


for num_valid_encodings in num_ones_num_valid_encoding:
	print(num_valid_encodings)

# fig = plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# # plt.show()
# fig.savefig("output.png")

# Results:
# Num ones: [2,3,4,5,6]
# intermediate_dim: [2,3,4,5,6,7,8,9,10]
# Try using K step validation to estimate parameters
# [3512, 2886, 5580, 8880, 10000, 10000, 10000, 10000, 10000]
# [0, 724, 2154, 6720, 7731, 7972, 9752, 10000, 10000]
# [0, 970, 2495, 3735, 5995, 9414, 10000, 10000, 10000]
# [0, 0, 2729, 3552, 4019, 8589, 8401, 10000, 9954]
# [0, 235, 4285, 3164, 3316, 3742, 10000, 10000, 10000]
