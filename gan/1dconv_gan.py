from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, UpSampling1D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.convolutional import Conv1D, MaxPooling1D

from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.ld_data_load as ldl

def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*15))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((15, 128), input_shape=(128*15,)))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(64, 5, padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(1, 5, padding='same'))
    model.add(Activation('tanh'))
    return model

def discriminator_model():
    model = Sequential()
    model.add(Conv1D(64, 5, padding='same', 
        input_shape=(120,1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, 5))
    model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
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

N =   100000
N_TR = 90000

def get_sample(num_samples, sample_dim, num_ones):
    index = np.random.randint(sample_dim, size=(num_samples,num_ones))
    zeros_val = np.zeros((num_samples,sample_dim))
    # print(index)
    for i,index_val in enumerate(index):
        # print(i,index_val)
        zeros_val[i,index_val] = 1
        # print(zeros_val[i])
    print(zeros_val.shape)
    print(zeros_val[:10])
    print(index[:50])
    return zeros_val


def train(BATCH_SIZE):
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train = (X_train.astype(np.float32) - 127.5)/127.5
    # X_train = X_train[:, :, :, None]
    # X_test = X_test[:, :, :, None]

    x_all = ldl.get_data(True, N)
    x_all = ldl.split_vector(x_all)
    print(x_all.shape)
    X_train = x_all[:N_TR]
    X_test = x_all[N_TR:]

    x_all = get_sample(N, 120, 2)
    X_train = x_all

    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    d.summary()
    d_on_g.summary()

    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = np.expand_dims(\
                X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE], axis=2)
            generated_images = g.predict(noise, verbose=0)
            # print(generated_images.shape, image_batch.shape)
            if index % 20 == 0:
                X1 = np.copy(generated_images)
                np.squeeze(X1)
                X1[X1<0.5] = 0
                X1[X1>=0.5] = 1
                print(X1[:10])
                print(np.sum(X1[:10], axis=1))
                print(np.sum(image_batch[:10], axis=1))
                # image = combine_images(generated_images)
                # image = image*127.5+127.5
                # Image.fromarray(image.astype(np.uint8)).save(
                #     "../../data/gan/images/"+str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE

            d_loss = d.train_on_batch(X, y)

            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            while(g_loss > 0.75*d_loss):
                g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            
            d.trainable = True
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
