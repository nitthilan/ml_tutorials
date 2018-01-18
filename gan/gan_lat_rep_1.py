
import os
import os, sys

import random
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
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bayesian_helpers as bh
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import link_distribution_16 as ld16
ld = ld16.LinkDistribution()


import pickle

import data_access as da

INPUT_DIM = 480
INTERMEDIATE_DIM = 1000
NOISE_DIM = 10

NUM_DATA = 100000
BATCH_SIZE = 10000
EPOCHS = 500
V_FREQ = 10
TRAIN_OFFSET = int(0.9*NUM_DATA)
IS_STORED = True



# List of things to try:
#   - https://github.com/soumith/ganhacks
# increase the number of data points
# increase the depth of the network
# manipulate the latent representation size - try finding a optimal
# Change the values between -1 to 1
# Validate the generated discrete structure validity like 
#   sum, connectivity, num links of each type etc
#   how does vae compare to gan

# - Find a optimal model for discriminator/generator
# - Try spliting the input into 4 regions and generate four different graphs
# - Try a different representation, probably a 16x16 representation
# - Try binary input of using mnist data set
# - Modify the algorithm

# Links:
# - https://github.com/nightrome/really-awesome-gan
#  - https://casmls.github.io/general/2017/04/13/gan.html
# - https://arxiv.org/abs/1702.07983v1
# https://github.com/zhangqianhui/AdversarialNetsPapers

# - https://www.tensorflow.org/programmers_guide/datasets
# - https://www.tensorflow.org/api_docs/python/tf/train/QueueRunner
# - https://stackoverflow.com/questions/34293714/can-i-measure-the-execution-time-of-individual-operations-with-tensorflow/37774470#37774470

# - https://www.reddit.com/r/MachineLearning/comments/4r3pjy/variational_autoencoders_vae_vs_generative/
# - https://arxiv.org/pdf/1512.09300.pdf - AAE (VAE+GAN)
# - http://people.ee.duke.edu/~lcarin/Yunchen9.30.2016.pdf - introduction to different GANS
# - http://cs231n.stanford.edu/reports/2015/pdfs/jgauthie_final_report.pdf
# - http://www.offconvex.org/2017/03/15/GANs/ - read1

# - https://myurasov.github.io/2017/09/24/wasserstein-gan-keras.html
# - http://shaofanlai.com/post/9

# Debugging using
# - https://medium.com/chakki/debug-keras-models-with-tensorflow-debugger-2b68b8e38370
# - https://www.tensorflow.org/programmers_guide/debugger
# - https://github.com/raghakot/keras-vis

# Three important controlling factors in NN:
# - Depth more important than breath of layers
# - Learning rate has to be modified or decreased as and when needed to increase learning rate
# - Batch normalisation layer seems to be very important
#   - https://github.com/fchollet/keras/issues/1802
#   - https://keras.io/layers/normalization/


# List of things to try:
#   - https://github.com/soumith/ganhacks
# increase the number of data points
# increase the depth of the network
# manipulate the latent representation size - try finding a optimal
# Change the values between -1 to 1
# Validate the generated discrete structure validity like 
#   sum, connectivity, num links of each type etc
#   how does vae compare to gan

# Links:
# - https://github.com/nightrome/really-awesome-gan
#  - https://casmls.github.io/general/2017/04/13/gan.html
# - https://arxiv.org/abs/1702.07983v1
# https://github.com/zhangqianhui/AdversarialNetsPapers

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# http://www.faqs.org/faqs/ai-faq/neural-nets/part1/preamble.html
# https://www.tensorflow.org/tutorials/layers
# https://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde

 


data = da.get_data(IS_STORED)
print(data.shape)
# unique_rows = bh.unique_rows(data)
# print(np.sum(unique_rows), unique_rows.shape)
# print(unique_rows)
# print(all(unique_rows))
# unique_rows[100] = False
# print(all(unique_rows))
  
# Dimentionality reduction:
# - tSNE, 
# - http://scikit-learn.org/stable/modules/manifold.html

def get_generative_1():
    gin = Input(shape=[NOISE_DIM])
    x = Dense(50)(gin)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(50)(gin)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(100)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(100)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(100)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(200)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(200)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(200)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    # x = Dense(480)(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.3)(x)
    # x = Dense(480)(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.3)(x)
    gout = Dense(INPUT_DIM, activation='tanh')(x)
    g = Model(gin, gout)
    return g, gin, gout

def get_generative(G_in, dense_dim=INTERMEDIATE_DIM, input_dim=INPUT_DIM, lr=1e-4):
    x = Dense(dense_dim, activation='relu')(G_in)
    x = Dense(dense_dim, activation='relu')(x)
    x = Dense(dense_dim, activation='relu')(x)
    x = Dense(dense_dim, activation='relu')(x)
    x = Dense(dense_dim, activation='relu')(x)
    x = Dense(dense_dim, activation='relu')(x)
    G_out = Dense(input_dim, activation='tanh')(x)
    G = Model(G_in, G_out)
    # opt = SGD(lr=lr)
    # G.compile(loss='binary_crossentropy', optimizer=opt)
    return G, G_out

with tf.device('/gpu:1'):
    # G_in = Input(shape=[NOISE_DIM])
    # G, G_out = get_generative(G_in)
    g, gin, gout = get_generative_1()
    g.summary()


def get_discriminative(D_in, lr=1e-3, drate=.25, n_channels=50, conv_sz=5, leak=.2):
    # x = Reshape((-1, 1))(D_in)
    # x = Conv1D(n_channels, conv_sz, activation='relu')(x)
    x = Dense(INTERMEDIATE_DIM, activation='relu')(D_in)
    x = Dense(INTERMEDIATE_DIM, activation='relu')(x)
    # x = Dense(INTERMEDIATE_DIM, activation='relu')(x)
    # x = Dense(INTERMEDIATE_DIM, activation='relu')(x)
    # x = Dense(INTERMEDIATE_DIM, activation='relu')(x)
    x = Dropout(drate)(x)
    # x = Flatten()(x)
    x = Dense(n_channels)(x)
    D_out = Dense(2, activation='sigmoid')(x)
    D = Model(D_in, D_out)
    D.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr))
    return D, D_out

def get_discriminative_1():
    din = Input(shape=[INPUT_DIM])
    x = Dense(480)(din)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(480)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(240)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(240)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(120)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(120)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(60)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(30)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(10)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    dout = Dense(2, activation='sigmoid')(x)
    d = Model(din, dout)
    return d, dout
with tf.device('/gpu:1'):
    # D_in = Input(shape=[INPUT_DIM])
    # D, D_out = get_discriminative(D_in)
    d, dout = get_discriminative_1()
    d.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3))

d.summary()

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
        
# def make_gan(gin, g, d):
#     # set_trainability(D, False)
#     x = G()
#     # D_GAN = Model(D.layers[0], D.layers[-1])
#     set_trainability(D, False)
#     GAN_out = D(x)
#     GAN = Model(GAN_in, GAN_out)
#     GAN.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3))
#     return GAN, GAN_out

with tf.device('/gpu:1'):
    dout = d(gout)
    gan = Model(gin, dout)
    set_trainability(d, False)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-2))

    # GAN_in = Input([NOISE_DIM])
    # GAN, GAN_out = make_gan(GAN_in, G, D)
 
gan.summary()


def sample_noise(n_samples, noise_dim=NOISE_DIM):
    # X = np.random.uniform(0, 1, size=[n_samples, noise_dim])
    X = np.random.normal(0, 1, size=[n_samples, noise_dim])
    return X

# def sample_fake_data(G, noise_dim=NOISE_DIM, n_samples=NUM_DATA):

#     XN = G.predict(XN_noise)
#     X = np.concatenate((XT, XN))
#     # y = np.zeros(2*n_samples)
#     y = np.zeros((2*n_samples, 2))
#     y[:n_samples, 1] = 1
#     y[n_samples:, 0] = 1
#     return X, y

# def pretrain(G, D, noise_dim=NOISE_DIM, n_samples=NUM_DATA, batch_size=BATCH_SIZE):
#     X, y = sample_data_and_gen(G, n_samples=n_samples, noise_dim=noise_dim)
#     set_trainability(D, True)
#     D.fit(X, y, epochs=1, batch_size=batch_size)

# pretrain(G, D)

def train(gan, g, d, epochs=EPOCHS, n_samples=NUM_DATA, noise_dim=NOISE_DIM, batch_size=BATCH_SIZE, verbose=False, v_freq=V_FREQ):
    d_loss = []
    g_loss = []

    XT = da.get_data(IS_STORED)
    XT[XT==0] = -1

    # y = np.zeros(2*NUM_DATA)
    Y_DIS = np.zeros((2*NUM_DATA, 2))
    Y_DIS[:NUM_DATA, 1] = 1
    Y_DIS[NUM_DATA:, 0] = 1
    Y_GEN = np.zeros((2*NUM_DATA, 2))
    Y_GEN[:NUM_DATA, 0] = 1
    Y_GEN[NUM_DATA:, 1] = 1

    # set_trainability(d, True)
    # d.fit(X, y,epochs=1, batch_size=BATCH_SIZE)

    
    e_range = range(epochs)
    # if verbose:
    #     e_range = tqdm(e_range)
    
    for epoch in e_range:
        set_trainability(d, True)
        X_Noise = sample_noise(NUM_DATA)
        X_fake = g.predict(X_Noise)
        X_FULL = np.concatenate((XT, X_fake))
        d.fit(X_FULL, Y_DIS,epochs=10, batch_size=BATCH_SIZE)

        y_pred = d.predict(X_FULL)

        print(y_pred[NUM_DATA-10:NUM_DATA+10])
        print(X_fake[:3])
        # print(X_fake[:3])
        
        set_trainability(d, False)
        gan.fit(X_Noise, Y_GEN[NUM_DATA:],epochs=100, batch_size=BATCH_SIZE)

        
        # print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".\
        #     format(epoch + 1, g_loss[-1], d_loss[-1]))
        X_fake_cpy = X_fake.copy()
        X_fake_cpy[X_fake<=0] = 0
        X_fake_cpy[X_fake>0] = 1
        sum_vector = np.sum(X_fake_cpy, axis=1)
        print("Sum vector ", sum_vector[:40])
        ld16.validate_sw_distribution(X_fake_cpy[:40])
        # ld16.validate_sw_distribution(X_true[:40])
        # print(X_fake_cpy[:40])




            
        

        # X, y = sample_noise(G, n_samples=n_samples, noise_dim=noise_dim)
        # set_trainability(D, False)
        # # set_trainability(G, True)
        # num_epocs = 1
        # for _ in range(num_epocs):
        #     for i in range(0, X.shape[0], BATCH_SIZE):
        #         # print(i)
        #         g_loss.append(GAN.train_on_batch(X[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]))
        
        # if verbose and (epoch + 1) % v_freq == 0:
        #     print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
        #     # print("Predicted Vector ", n_samples, X1.shape, X1[NUM_DATA:NUM_DATA+10])
        #     X1[X1<0.5] = 0
        #     X1[X1>=0.5] = 1
        #     sum_vector = np.sum(X1, axis=1)
        #     print("Sum vector ", sum_vector[NUM_DATA-10:NUM_DATA+10])
        #     ld16.validate_sw_distribution(X1[NUM_DATA-10:NUM_DATA+10])


        # if (epoch + 1) % epochs == 0 :
        #     X1[X1<0.5] = 0
        #     X1[X1>=0.5] = 1
        #     sum_vector = np.sum(X1, axis=1)
        #     print("Sum vector ", sum_vector[NUM_DATA:NUM_DATA+40])
        #     ld16.validate_sw_distribution(X1[NUM_DATA:NUM_DATA+40])

    return d_loss, g_loss

# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)
d_loss, g_loss = train(gan, g, d, verbose=True)
