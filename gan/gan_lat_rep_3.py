
import os
import os, sys

import random as rd
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm_notebook as tqdm
import keras.backend as K
from tensorflow.python import debug as tf_debug

from keras.models import Model
from keras.layers import Input, Reshape
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.layers.advanced_activations import LeakyReLU

from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import tensorflow as tf
from keras.models import Sequential
from keras.regularizers import l2

# https://github.com/soumith/ganhacks#authors
# https://deeplearning4j.org/generative-adversarial-network
# https://arxiv.org/abs/1605.07725 - Adversarial Training Methods for Semi-Supervised Text Classification
# https://arxiv.org/pdf/1606.03498.pdf - Improved training for GAN
# https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners
# one sided label smoothing
# https://github.com/willwhitney/gan-article/blob/master/gans.md
# Problems: GAN generator generating the same output for all inputs

# Normalise inputs between -1 to 1
#	- Subtract mean and variance


# https://openreview.net/pdf?id=BJcAWaeCW
# https://www.youtube.com/watch?v=A3ekFcZ3KNw - david silvers alphago one

# Deep Learing Book Ian GoodFellow:
# - http://www.deeplearningbook.org/
# - deep learning by goodfellow pdf
# - Ian GoodFellows lecture series: http://www.deeplearningbook.org/lecture_slides.html
# - https://github.com/janishar/mit-deep-learning-book-pdf

# https://arxiv.org/pdf/1701.00160.pdf - NIPS 2016 Tutorial: Generative Adversarial Networks
# https://arxiv.org/pdf/1511.06434.pdf - UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS

# http://papers.nips.cc/paper/7010-learning-active-learning-from-data.pdf



# Some old references:
# https://arxiv.org/pdf/1312.6114.pdf
# https://arxiv.org/pdf/1606.05908.pdf
# http://kvfrans.com/generative-adversial-networks-explained/
# http://kvfrans.com/variational-autoencoders-explained/
# https://www.google.co.in/search?q=autoencoder+variational&rlz=1C5CHFA_enIN713IN714&oq=auto+encoder+var&aqs=chrome.1.69i57j0l2.14205j0j7&sourceid=chrome&ie=UTF-8

# http://www.jmlr.org/papers/volume11/ganchev10a/ganchev10a.pdf
# https://www.reddit.com/r/MachineLearning/comments/6hi5a4/r_170604223_adversarially_regularized/
# https://www.google.co.in/search?q=Encoder-decoder+architectures+especially+over+discrete+structures&rlz=1C5CHFA_enIN713IN714&oq=Encoder-decoder+architectures+especially+over+discrete+structures&aqs=chrome..69i57&sourceid=chrome&ie=UTF-8
# http://www.cs.dartmouth.edu/~qliu/PDF/ksd_short.pdf
# https://arxiv.org/pdf/1602.03253v2.pdf
# https://arxiv.org/pdf/1608.04471.pdf


# N - Num inputs
# D - Max dimension of input vector
# K - num non-zero inputs
def get_k_ones(N, D, K):
	pos_sam = np.zeros((N, D))
	# pos_sam = -1*np.ones((N, D))
	for i in range(N):
		pos_sam[i,rd.sample(range(D), K)] = 1

	mean = np.sum(pos_sam[0])*1.0/D
	pos_sam = pos_sam - mean
	noise = np.random.normal(loc=0.0, scale=0.001, size=(N, D))
	pos_sam += noise
	return pos_sam


def get_discriminator(D):
	LRELU_ALPHA = 0.02
	reg=lambda: l2(0.01)
	din = Input(shape=[D], name="input_dis_0")
	x = Dense(2*D,
		name="dense_dis_0",
		kernel_initializer='random_uniform',
        bias_initializer='zeros',
        W_regularizer=reg()
        )(din)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=LRELU_ALPHA)(x)

	# x = Dense(DIS_INTER_DIM)(x)
	# x = BatchNormalization()(x)
	# x = LeakyReLU(alpha=DIS_LRELU_ALPHA)(x)
	dout = Dense(1,
		name="dense_dis_1",
		kernel_initializer='random_uniform',
        bias_initializer='zeros',
        W_regularizer=reg(),
        activation='sigmoid')(x)
	d = Model(din, dout)
	return d, dout

def get_generator(L, D):
	LRELU_ALPHA = 0.2
	reg=lambda: l2(0.01)
	gin = Input(shape=[L], name="input_0")
	x = Dense(20,
		name="dense_0",
		kernel_initializer='random_uniform',
        bias_initializer='zeros',
        W_regularizer=reg()
        )(gin)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=LRELU_ALPHA)(x)
	# x = Activation('relu')(x)

	x = Dense(40,
		name="dense_1",
		kernel_initializer='random_uniform',
        bias_initializer='zeros',
        W_regularizer=reg()
        )(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=LRELU_ALPHA)(x)
	# x = Activation('relu')(x)

	x = Dense(80,
		name="dense_2",
		kernel_initializer='random_uniform',
        bias_initializer='zeros',
        W_regularizer=reg()
        )(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=LRELU_ALPHA)(x)
	# x = Activation('relu')(x)

	x = Dense(160,
		name="dense_3",
		kernel_initializer='random_uniform',
        bias_initializer='zeros',
        W_regularizer=reg()
        )(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=LRELU_ALPHA)(x)
	# x = Activation('relu')(x)

	# x = Dense(160,
	# 	kernel_initializer='truncated_normal',
 #        bias_initializer='zeros',
 #        W_regularizer=reg())(x)
	# x = BatchNormalization()(x)
	# x = LeakyReLU(alpha=LRELU_ALPHA)(x)
	# # x = Activation('relu')(x)

	# x = Dense(160,
	# 	kernel_initializer='truncated_normal',
 #        bias_initializer='zeros',
 #        W_regularizer=reg())(x)
	# x = BatchNormalization()(x)
	# x = LeakyReLU(alpha=LRELU_ALPHA)(x)
	# # x = Activation('relu')(x)

	# x = Dense(160,
	# 	kernel_initializer='truncated_normal',
 #        bias_initializer='zeros',
 #        W_regularizer=reg())(x)
	# x = BatchNormalization()(x)
	# x = LeakyReLU(alpha=LRELU_ALPHA)(x)
	# # x = Activation('relu')(x)


	# x = Dense(160,
	# 	kernel_initializer='truncated_normal',
 #        bias_initializer='zeros',
 #        W_regularizer=reg())(x)
	# x = BatchNormalization()(x)
	# # x = LeakyReLU(alpha=LRELU_ALPHA)(x)
	# # x = Activation('relu')(x)

	# gout = Dense(D, activation='tanh')(x)
	# gout = Dense(D, activation='sigmoid')(x)
	gout = Dense(D,
		name="dense_4",
		kernel_initializer='random_uniform',
        bias_initializer='zeros', 
        W_regularizer=reg(),
        activation='tanh')(x)

	g = Model(gin, gout)
	return g, gin, gout 

def gen_gan_network(gen_net, dis_net, LR):
	gan = Sequential()
	gan.add(gen_net)
	dis_net.trainable = False
	gan.add(dis_net)
	return gan

earlyStopping=EarlyStopping(monitor='val_loss', 
		patience=10, verbose=0, mode='auto',
		min_delta=0.0001)
reduceonplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
	patience=4, verbose=0, mode='auto', 
	epsilon=0.0001, cooldown=0, min_lr=0)
csv_logger = CSVLogger('training.log')
tensorboard_dis = TensorBoard(log_dir='./logs_dis', histogram_freq=1, batch_size=32, 
	write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)



L = 2
D = 10
POS_K = 2


def train_dis_network(dis_net, x_neg, x_pos):
	BATCH_SIZE = 100
	NUM_EPOCHS = 1
	LR = 0.00005

	N1 = len(x_neg)
	N2 = len(x_pos)
	y_full = np.zeros(N1+N2)
	y_full[:N1] = 0
	y_full[N1:] = 1
	x_full = np.concatenate((x_neg, x_pos))
	print(x_full[:2], x_full[N1:N1+2])
	print(y_full[:2], y_full[N1:N1+2])

	print("Training Discriminator ")
	
	dis_net.fit(x_full, y_full, epochs=NUM_EPOCHS, \
					batch_size=BATCH_SIZE, 
					# shuffle=True,
					validation_split=.00001,
					callbacks=[
					# tensorboard_dis,
					# 	earlyStopping, 
					# 	reduceonplateau,
					# 	csv_logger
					]
					)

	x_pred = np.concatenate((x_pos[:10], x_neg[:10]))
	y_pred = dis_net.predict(x_pred)
	print("Disc Validation ",y_pred)

	return

def train_gen_network(gan_net, gen_net, dis_net, x_laten, num_iter):
	BATCH_SIZE = 100
	NUM_EPOCHS = 5
	LR = 0.00005

	tensorboard_gen = TensorBoard(log_dir='./logs/logs_gen_'+str(num_iter), histogram_freq=1, batch_size=32, 
		write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)


	print("Training GAN ")
	gan_net.fit(x_laten, 0.9*np.ones(len(x_laten)), 
				epochs=NUM_EPOCHS,
				batch_size=BATCH_SIZE, 
				# shuffle=True,
				validation_split=.00001,
				callbacks=[
				tensorboard_gen,
				# 	earlyStopping, 
				# 	reduceonplateau,
				# 	csv_logger
				]
				)

	x_new  = np.random.normal(0, 1, size=[10, L])
	x_new[x_new <= -1] = -1
	x_new[x_new >=  1] =  1
	x_pred = np.concatenate((x_laten[:10], x_new))
	y_pred = gen_net.predict(x_pred)
	y_pred_dis = dis_net.predict(y_pred)
	y_pred_gan = gan_net.predict(x_pred)
	print(x_pred)
	i = 0
	for y_pred_val in y_pred:
		print(y_pred_gan[i,:], y_pred_dis[i,:])
		print(y_pred_val[:])
		y_pred_1 = np.copy(y_pred_val)
		y_pred_1[y_pred_val < 0] = -1
		y_pred_1[y_pred_val >= 0] = 1
		print(y_pred_1[:])
		i += 1
	print("Sum ", np.sum(y_pred, axis=1))

	return

# def pre_train_gen(gen_net):
# 	return
def train_gen_using_dis():
	N = 100000
	BATCH_SIZE = 1000
	NUM_EPOCHS = 400
	LR = 0.00005
	DLR = 0.0005

	gen_net, gen_in, gen_out = get_generator(L, D)
	dis_net, dis_out = get_discriminator(D)

	dis_net.compile(loss='binary_crossentropy', 
		optimizer=Adam(lr=DLR))

	# dis_net.load_weights("discriminator.h5")
	# gen_net.load_weights("generator.h5")
	gan_net = gen_gan_network(gen_net, dis_net, LR)
	gan_net.compile(loss='binary_crossentropy',
		optimizer=Adam(lr=LR))
	# optimizer = SGD(lr=0.00005, momentum=0.9, nesterov=True)

	# Negative Samples
	NEG_N = 10000
	x_neg = np.random.normal(0, 1, size=[NEG_N, D])
	NEG_K = np.asarray([1,3,4,5,6])
	for i in range(len(NEG_K)):
		x_neg_i = get_k_ones(NEG_N, D, NEG_K[i])
		x_neg = np.concatenate((x_neg_i, x_neg))

	# Positive Samples
	x_pos = get_k_ones(N, D, 2)

	# dis_net.trainable = True
	# train_dis_network(dis_net, x_neg, x_pos)

	# Latent representation
	x_laten_full = np.random.normal(0, 1, size=[N, L])
	x_laten_full[x_laten_full <= -1] = -1
	x_laten_full[x_laten_full >=  1] =  1
	x_neg_full = gen_net.predict(x_laten_full)

	# Positive Samples
	x_pos_full = get_k_ones(N, D, 2)

	for num_iter in range(50):
		dis_net.trainable = True
		# train_discriminator_network(dis_net, gen_net, 7)
		# Latent representation
		x_laten = np.random.normal(0, 1, size=[N, L])
		x_laten[x_laten <= -1] = -1
		x_laten[x_laten >=  1] =  1
		x_neg = gen_net.predict(x_laten)
		x_neg_full = np.concatenate((x_neg_full, x_neg))
		x_laten_full = np.concatenate((x_laten_full, x_laten))

		# Positive Samples
		x_pos = get_k_ones(N, D, 2)
		x_pos_full = np.concatenate((x_pos_full, x_pos))
		# x_dis_neg = np.concatenate((x_neg_lat, x_neg))
		train_dis_network(dis_net, x_neg_full, x_pos_full)


		# gen_net.load_weights("generator.h5")		
		dis_net.trainable = False
		# train_gan_network(gan_net, gen_net, dis_net)
		train_gen_network(gan_net, gen_net, dis_net, x_laten_full, num_iter)
	return






def train_discriminator_network(dis_net, gen_net, N1):
	N = 100000
	
	NEG_K = np.asarray([1,3,4,5,6])#,7,8,9,10,11,12])
	BATCH_SIZE = 1000
	NUM_EPOCHS = 3
	LR = 0.00005

	y_full = np.zeros(2*N)
	y_full[:N] = 0
	y_full[N:] = 1
	# print(x_full[0], y_full[0])
	# print(x_full[N], y_full[1])

	print("Training Discriminator ", N, D, L, K)

	if(N1 != 0):
		num_iter = N1
	else:
		num_iter = len(NEG_K)

	for i in range(num_iter):
		x_pos = get_k_ones(N, D, POS_K)
		print("x_pos ", i, x_pos[:5])
		if(N1 == 0):
			x_neg_1 = get_k_ones(N/2, D, NEG_K[i])
		else:
			x_laten = np.random.normal(0, 1, size=[N/2, L])
			x_laten[x_laten <= -1] = -1
			x_laten[x_laten >=  1] =  1
			x_neg_1 = gen_net.predict(x_laten)
		x_neg_2 = np.random.uniform(0, 1, size=[N/2, D])
		x_neg = np.concatenate((x_neg_2, x_neg_1))
		x_neg[x_neg <= -1] = -1
		x_neg[x_neg >=  1] =  1
	
		x_full = np.concatenate((x_pos, x_neg))

		dis_net.fit(x_full, y_full, epochs=NUM_EPOCHS, \
					batch_size=BATCH_SIZE, 
					# shuffle=True,
					# validation_split=.1,
					callbacks=[
					tensorboard
					# 	earlyStopping, 
					# 	reduceonplateau,
					# 	csv_logger
					]
					)
		x_full = np.concatenate((x_pos[:10], x_neg[:10]))
		y_pred = dis_net.predict(x_full)
		print("Discriminator ",y_pred)
		dis_net.save_weights("discriminator.h5")

	x_pos = get_k_ones(10, D, 2)
	x_neg = get_k_ones(10, D, 5)
	x_full = np.concatenate((x_pos, x_neg))
	y_pred = dis_net.predict(x_full)
	print("Disc Validation ",y_pred)

	return

def train_gan_network(gan_net, gen_net, dis_net):
	N = 100000
	BATCH_SIZE = 1000
	NUM_EPOCHS = 10
	LR = 0.00005

	print("Training GAN ")

	for _ in range(5):
		# x_pos = get_k_ones(N, D, POS_K)
		# print("x_pos ", _, x_pos[:5])
		x_laten = np.random.normal(0, 1, size=[N, L])
		x_laten[x_laten <= -1] = -1
		x_laten[x_laten >=  1] =  1
		# x_neg = gen_net.predict(x_lat)
		# Pos and neg samples swapped for GAN training
		# x_full = np.concatenate((x_neg,x_pos))

		gan_net.fit(x_laten, np.ones(N), 
					epochs=NUM_EPOCHS,
					batch_size=BATCH_SIZE, 
					# shuffle=True,
					# validation_split=.1,
					callbacks=[
					tensorboard
					# 	earlyStopping, 
					# 	reduceonplateau,
					# 	csv_logger
					]
					)
		x_new  = np.random.normal(0, 1, size=[10, L])
		x_new[x_new <= -1] = -1
		x_new[x_new >=  1] =  1
		x_pred = np.concatenate((x_laten[:10], x_new))
		y_pred = gen_net.predict(x_pred)
		y_pred_dis = dis_net.predict(y_pred)
		y_pred_gan = gan_net.predict(x_pred)
		print(x_pred)
		i = 0
		for y_pred_val in y_pred:
			print(y_pred_gan[i], y_pred_dis[i])
			print(y_pred_val)
			i += 1
		print("Sum ", np.sum(y_pred, axis=1))
		gen_net.save_weights("generator.h5")

	return

















def train_generator_network():
	N = 100000
	L = 10#50
	D = 100
	POS_K = 2
	NEG_K = 3
	BATCH_SIZE = 100
	NUM_EPOCHS = 100
	LR = 0.0005

	y_full = get_k_ones(N, D, POS_K)
	x_full = np.random.uniform(0, 1, size=[N, L])
	print(x_full[:10])

	gen_net, gen_in, gen_out = get_generator(L, D)
	gen_net.compile(loss='binary_crossentropy', 
		optimizer=Adam(lr=LR))
	gen_net.fit(x_full, y_full, epochs=100, \
				batch_size=BATCH_SIZE, 
				shuffle=True,
				validation_split=.1,
				# callbacks=[
				# 	earlyStopping, 
				# 	reduceonplateau,
				# 	csv_logger]
				)
	x_new  = np.random.uniform(0, 1, size=[10, L])
	x_pred = np.concatenate((x_full[:10], x_new))
	y_pred = gen_net.predict(x_pred)
	print(x_pred)
	print(y_pred)
	print(np.sum(y_pred, axis=1))
	gen_net.save_weights("generator.h5")
	return gen_net, gen_in, gen_out

def train_gen_net():
	N = 100000
	L = 10#50
	D = 100
	POS_K = 2
	NEG_K = 3
	BATCH_SIZE = 100
	NUM_EPOCHS = 100
	LR = 0.0005#005

	y_full = get_k_ones(N, D, POS_K)
	gen_net, gen_in, gen_out = get_generator(L, D)
	gen_net.compile(loss='binary_crossentropy', 
		optimizer=Adam(lr=LR))

	# gen_net.load_weights("generator.h5")

	print("Try no ", 1)

	for _ in range(10):
		x_full = np.random.uniform(0, 1, size=[N, L])
		print("Num iter ", _, x_full[:10])
		gen_net.fit(x_full, y_full, epochs=100, \
					batch_size=BATCH_SIZE, 
					shuffle=True,
					validation_split=.1,
					# callbacks=[
					# 	earlyStopping, 
					# 	reduceonplateau,
					# 	csv_logger]
					)
		x_new  = np.random.uniform(0, 1, size=[10, L])
		x_pred = np.concatenate((x_full[:10], x_new))
		y_pred = gen_net.predict(x_pred)
		print(x_pred)
		for y_pred_val in y_pred:
			print(y_pred_val)
		print(np.sum(y_pred, axis=1))
		gen_net.save_weights("generator.h5")
	return gen_net, gen_in, gen_out

def train_gan_network_2():
	L = 10
	D = 100

	dis_net, dis_out = get_discriminator(D)
	gen_net, gen_in, gen_out = train_generator_network()
	train_discriminator_network(dis_net)
	gan = gen_gan_network(gen_net, dis_net)

	N = 100000
	BATCH_SIZE = 100
	NUM_EPOCHS = 100

	for _ in range(10):
		x_full = np.random.uniform(0, 1, size=[N, L])
		y_full = np.ones(N)

		dis_net.trainable = False
		gan.fit(x_full, y_full, epochs=NUM_EPOCHS, \
					batch_size=BATCH_SIZE, 
					shuffle=True,
					validation_split=.1,
					# callbacks=[
					# 	earlyStopping, 
					# 	reduceonplateau,
					# 	csv_logger]
					)

		dis_net.trainable = True
		train_discriminator_network(dis_net)
	return

def train_gan_network_1():
	N = 100000
	L = 10
	D = 100
	POS_K = 2
	NEG_K = 3
	BATCH_SIZE = 100
	NUM_EPOCHS = 100
	LR = 0.03

	x_pos = get_k_ones(N, D, POS_K)
	x_neg = get_k_ones(N, D, NEG_K)
	x_full = np.concatenate((x_pos, x_neg))

	y_full = np.zeros(2*N)
	y_full[:N] = 0
	y_full[N:] = 1
	# print(x_full[0], y_full[0])
	# print(x_full[N], y_full[1])
	dis_net, dout = get_discriminator(D)
	dis_net.compile(loss='binary_crossentropy', 
		optimizer=Adam(lr=LR))
	dis_net.fit(x_full, y_full, epochs=20, \
				batch_size=BATCH_SIZE, 
				shuffle=True,
				validation_split=.1,
				# callbacks=[
				# 	earlyStopping, 
				# 	reduceonplateau,
				# 	csv_logger]
				)


	gen_net, gin, gout = get_generator(L, D)
	gan = Sequential()
	gan.add(gen_net)
	dis_net.trainable = False
	gan.add(dis_net)

	gan.compile(loss='binary_crossentropy',
		optimizer=Adam(lr=0.3))

	x_full = np.random.normal(0, 1, size=[N, L])
	y_full = np.ones(N)

	for _ in range(NUM_EPOCHS):
		g_loss = gan.train_on_batch(x_full, y_full)
		print("Gen loss ", _, g_loss)
		x_pred = np.random.normal(0, 1, size=[2, L])
		y_pred = gen_net.predict(x_pred)
		print(y_pred)
	# gan.fit(x_full, y_full, epochs=NUM_EPOCHS, \
	# 			batch_size=BATCH_SIZE, 
	# 			shuffle=True,
	# 			validation_split=.1,
				# callbacks=[
				# 	earlyStopping, 
				# 	reduceonplateau,
				# 	csv_logger]
				# )
	return

# D = 100
# dis_net, dis_out = get_discriminator(D)
# train_discriminator_network(dis_net)
train_gen_using_dis()
# train_generator_network()
# train_gan_network()
# train_gen_net()