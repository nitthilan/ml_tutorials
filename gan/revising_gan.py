from keras.models import Sequential
from keras.layers import Dense, UpSampling1D, Conv1D, Input
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, UpSampling1D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.convolutional import Conv1D, MaxPooling1D

from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math
from keras import regularizers


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.ld_data_load as ldl

# stratified sampling
# Manifold Alignment
# detecting large gradients in keras

# N=100000
# N_TR=90000

num_samples = 100000
sample_dim = 50
noise_dim = 10
num_ones = 2

def get_sample(num_samples, sample_dim, num_ones):
    index = np.random.randint(sample_dim, size=(num_samples,num_ones))
    zeros_val = -1*np.ones((num_samples,sample_dim))
    # print(index)
    for i,index_val in enumerate(index):
        # print(i,index_val)
        zeros_val[i,index_val] = 1
        # print(zeros_val[i])
    # print(zeros_val.shape)
    # print(zeros_val[:10])
    # print(index[:50])
    return zeros_val


def gen_dense_model(noise_dim, sample_dim, num_layers):
	inter_dim = 100
	weight_decay = 0.0005
	model = Sequential()
	model.add(Dense(input_dim=noise_dim, units=inter_dim,
		kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dense(inter_dim,
		kernel_regularizer=regularizers.l2(weight_decay)))

	for i in range(num_layers):
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dense(sample_dim,
			kernel_regularizer=regularizers.l2(weight_decay)))

	model.add(Activation('tanh'))
	return model

def gen_conv_model(noise_dim, sample_dim):
	model = Sequential()
	# model.add(Input(shape=(noise_dim,)))
	model.add(Reshape((1, noise_dim), input_shape=(noise_dim,)))
	model.add(UpSampling1D(size=2))
	model.add(Conv1D(16, 3, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(UpSampling1D(size=2))
	model.add(Conv1D(32, 3, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(UpSampling1D(size=2))
	model.add(Conv1D(64, 3, padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('tanh'))

	model.add(Dense(sample_dim))
	model.add(Activation('tanh'))


	return model

def dis_dense_model(sample_dim):
	inter_dim = 100
	model = Sequential()
	model.add(Dense(input_dim=sample_dim, units=inter_dim))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	return model

def gen_dis_model(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def train(BATCH_SIZE):
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train = (X_train.astype(np.float32) - 127.5)/127.5
    # X_train = X_train[:, :, :, None]
    # X_test = X_test[:, :, :, None]

    # x_all = ldl.get_data(True, N)
    # x_all = ldl.split_vector(x_all)
    # print(x_all.shape)
    # X_train = x_all[:N_TR]
    # X_test = x_all[N_TR:]

    x_all = get_sample(num_samples, sample_dim, num_ones)
    X_train = x_all
    noise_list = np.random.uniform(-1, 1, size=(num_samples, noise_dim))


    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    d = dis_dense_model(sample_dim)
    # g = gen_dense_model(noise_dim, sample_dim, 100)
    g = gen_conv_model(noise_dim, sample_dim)
    d_on_g = gen_dis_model(g, d)
    # d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    # g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    d_optim = Adam(lr=0.0005, decay=1e-6)
    g_optim = Adam(lr=0.0005, decay=1e-6)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    d.summary()
    g.summary()
    d_on_g.summary()

    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = noise_list[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            # image_batch = np.expand_dims(\
            #     X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE], axis=2)
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            print(generated_images.shape, image_batch.shape)
            if index % 20 == 0:
                X1 = np.copy(generated_images)
                np.squeeze(X1)
                X1[X1<0] = -1
                X1[X1>=0] = 1
                # print(X1[:10])
                # print(generated_images[:10])
                print(np.sum(X1[:10], axis=1))
                print(np.sum(image_batch[:10], axis=1))
                # image = combine_images(generated_images)
                # image = image*127.5+127.5
                # Image.fromarray(image.astype(np.uint8)).save(
                #     "../../data/gan/images/"+str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE + np.random.normal(scale=0.25, size=2*BATCH_SIZE)
            d.trainable = True

            X_tr = X.copy() + np.random.normal(scale=0.25, size=X.shape)

            # print(X[:5])
            # print(X[-5:])
            print(y[:5])
            print(y[-5:])

            d_loss = d.train_on_batch(X_tr, y)
            print("batch %d d_loss : %f" % (index, d_loss))

            # d_loss = d.train_on_batch(image_batch, [1] * BATCH_SIZE)
            # print("batch %d d_loss : %f" % (index, d_loss))

            # d_loss = d.train_on_batch(generated_images, [0] * BATCH_SIZE)
            # print("batch %d d_loss : %f" % (index, d_loss))

            # noise = np.random.uniform(-1, 1, (BATCH_SIZE, noise_dim))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            # while(g_loss > d_loss):
            #     g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            #     print("batch %d g_loss : %f d_loss : %f" % (index, g_loss, d_loss))
            
            print("batch %d g_loss : %f" % (index, g_loss))
            # if index % 10 == 9:
            #     g.save_weights("../../data/gan/weights/generator", True)
            #     d.save_weights("../../data/gan/weights/discriminator", True)


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
