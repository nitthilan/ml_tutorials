
import os
import os, sys

import random
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm_notebook as tqdm
from keras.models import Model
from keras.layers import Input, Reshape
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import link_distribution_16 as ld16
ld = ld16.LinkDistribution()

import pickle

INPUT_DIM = 480
INTERMEDIATE_DIM = 200
NOISE_DIM = 10

NUM_DATA = 10000 #100000
BATCH_SIZE = 32
EPOCHS = 500
V_FREQ = 10
TRAIN_OFFSET = int(0.9*NUM_DATA)

# DUMP_FILE = "design_data_latent.pickle"
DUMP_FILE = "design_100000.npz"
IS_STORED = True



def save_data(x):

    np.savez(DUMP_FILE, x=x)
    
    # # Saving the objects:
    # with open(DUMP_FILE, 'w') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump(x, f)

    # return
def return_saved():
    return np.load(DUMP_FILE)['x']
    
    # # Saving the objects:
    # with open(DUMP_FILE, 'r') as f:  # Python 3: open(..., 'wb')
    #     return pickle.load(f)

def get_data(is_stored):
    if(not is_stored):
        rand_designs = ld.generate_n_random_feature(NUM_DATA)
        # rand_design_val = dummy_utility(rand_designs)
        save_data(rand_designs)
        # print(len(rand_designs), rand_desing_val)
        return rand_designs
    else:
        rand_designs = return_saved()
        # rand_designs[rand_designs<0.5] = -1
        return rand_designs[:NUM_DATA]

data = get_data(IS_STORED)

print(data.shape)

def get_generative(G_in, dense_dim=INTERMEDIATE_DIM, input_dim=INPUT_DIM, lr=1e-3):
    x = Dense(dense_dim)(G_in)
    x = Activation('tanh')(x)
    G_out = Dense(input_dim, activation='tanh')(x)
    G = Model(G_in, G_out)
    opt = SGD(lr=lr)
    G.compile(loss='binary_crossentropy', optimizer=opt)
    return G, G_out

G_in = Input(shape=[NOISE_DIM])
with tf.device('/gpu:1'):
    G, G_out = get_generative(G_in)
G.summary()

def get_discriminative(D_in, lr=1e-3, drate=.25, n_channels=50, conv_sz=5, leak=.2):
    x = Reshape((-1, 1))(D_in)
    x = Conv1D(n_channels, conv_sz, activation='relu')(x)
    x = Dropout(drate)(x)
    x = Flatten()(x)
    x = Dense(n_channels)(x)
    D_out = Dense(2, activation='sigmoid')(x)
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr)
    D.compile(loss='binary_crossentropy', optimizer=dopt)
    return D, D_out

D_in = Input(shape=[INPUT_DIM])
with tf.device('/gpu:1'):
    D, D_out = get_discriminative(D_in)
D.summary()

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
        
def make_gan(GAN_in, G, D):
    set_trainability(D, False)
    x = G(GAN_in)
    GAN_out = D(x)
    GAN = Model(GAN_in, GAN_out)
    GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer)
    return GAN, GAN_out

GAN_in = Input([NOISE_DIM])
GAN, GAN_out = make_gan(GAN_in, G, D)
GAN.summary()



def sample_noise(G, noise_dim=NOISE_DIM, n_samples=NUM_DATA):
    # X = np.random.uniform(0, 1, size=[n_samples, noise_dim])
    X = np.random.normal(0, 1, size=[n_samples, noise_dim])

    y = np.zeros((n_samples, 2))
    y[:, 1] = 1
    return X, y

def sample_data_and_gen(G, noise_dim=NOISE_DIM, n_samples=NUM_DATA):
    XT = get_data(IS_STORED)#sample_data(n_samples=n_samples)
    # XN_noise = np.random.uniform(0, 1, size=[n_samples, noise_dim])
    XN_noise = np.random.normal(0, 1, size=[n_samples, noise_dim])

    XN = G.predict(XN_noise)
    X = np.concatenate((XT, XN))
    y = np.zeros((2*n_samples, 2))
    y[:n_samples, 1] = 1
    y[n_samples:, 0] = 1
    return X, y

def pretrain(G, D, noise_dim=NOISE_DIM, n_samples=NUM_DATA, batch_size=BATCH_SIZE):
    X, y = sample_data_and_gen(G, n_samples=n_samples, noise_dim=noise_dim)
    set_trainability(D, True)
    D.fit(X, y, epochs=1, batch_size=batch_size)

pretrain(G, D)

def train(GAN, G, D, epochs=EPOCHS, n_samples=NUM_DATA, noise_dim=NOISE_DIM, batch_size=BATCH_SIZE, verbose=False, v_freq=V_FREQ):
    d_loss = []
    g_loss = []
    e_range = range(epochs)
    # if verbose:
    #     e_range = tqdm(e_range)
    for epoch in e_range:
        X, y = sample_data_and_gen(G, n_samples=n_samples, noise_dim=noise_dim)
        set_trainability(D, True)
        d_loss.append(D.train_on_batch(X, y))

        X1 = np.copy(X)
        


        X, y = sample_noise(G, n_samples=n_samples, noise_dim=noise_dim)
        set_trainability(D, False)
        g_loss.append(GAN.train_on_batch(X, y))
        if verbose and (epoch + 1) % v_freq == 0:
            print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
            print("Predicted Vector ", n_samples, X1.shape, X1[NUM_DATA:NUM_DATA+10])

        if (epoch + 1) % 500 == 0 :
            X1[X1<0.5] = 0
            X1[X1>=0.5] = 1
            sum_vector = np.sum(X1, axis=1)
            print("Sum vector ", sum_vector[NUM_DATA:NUM_DATA+40])
            ld16.validate_sw_distribution(X1[NUM_DATA:NUM_DATA+40])

    return d_loss, g_loss

d_loss, g_loss = train(GAN, G, D, verbose=True)
