
import sys,os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import random as rd

from keras import backend as K

from keras.callbacks import TensorBoard

from keras.optimizers import Adam
from keras.layers import BatchNormalization, PReLU
from keras.callbacks import ModelCheckpoint

def encoder_network(original_dim, intermediate_dim, latent_dim, num_layers):
    x = Input(shape=(original_dim,))

    layer_list = []
    for i in range(num_layers):
        layer_list.append(Dense(intermediate_dim, name="encoder_"+str(i)))
        layer_list.append(BatchNormalization())
        layer_list.append(PReLU())

    h = layer_list[0](x)
    for layer in layer_list[1:]:
        h = layer(h)

    z_mean = Dense(latent_dim, activation='linear', name="encoder_mean")(h)
    z_log_var = Dense(latent_dim, activation='softplus', name="encoder_log_var")(h)


    return x, z_mean, z_log_var

def decoder_network(z, intermediate_dim, original_dim, num_layers):

    layer_list = []
    for i in range(num_layers):
        layer_list.append(Dense(intermediate_dim, name="decoder_"+str(i)))
        layer_list.append(BatchNormalization())
        layer_list.append(PReLU())
    layer_list.append(Dense(original_dim, activation='linear', name="decoder_mean"))
    # layer_list.append(Dense(original_dim, activation='sigmoid', name="decoder_mean"))
    
    # we instantiate these layers separately so as to reuse them later
    h = layer_list[0](z)
    for layer in layer_list[1:]:
        h = layer(h)
    x_decoded_mean = h

    return layer_list, x_decoded_mean


def get_networks(original_dim, intermediate_dim, latent_dim, num_layers):

    def sampling(args):
        epsilon_std = 1.0

        z_mean, z_log_var = args
        shape_value = (K.shape(z_mean)[0], latent_dim)
        # shape_value = K.print_tensor(shape_value, message="shape_value is: ")
        epsilon = K.random_normal(shape=shape_value, mean=0.,
                                  stddev=epsilon_std)

        # z_mean = K.print_tensor(z_mean, message="z_mean is: ")
        # z_log_var = K.print_tensor(z_log_var, message="z_log_var is: ")

        z_sample = z_mean + K.exp(z_log_var/ 2) * epsilon
        # z_sample = epsilon
        # z_sample = K.print_tensor(z_sample, message="z_sample is: ")
        return z_sample

    x, z_mean, z_log_var = encoder_network(original_dim, \
        intermediate_dim, latent_dim, num_layers)
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    layer_list, x_decoded_mean = decoder_network(z, \
        intermediate_dim, original_dim, num_layers)
    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean):
            # xent_loss =  original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
            xent_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean) 
            # xent_loss = K.print_tensor(K.mean(xent_loss),  message="xent loss is: ")

            z_log_var_exp = K.exp(z_log_var)
            z_mean_sq = K.square(z_mean)
            # z_log_var_exp = K.print_tensor(z_log_var_exp, message="z_log_var avg: ")
            # z_mean_sq = K.print_tensor(z_mean_sq,  message="z_mean avg: ")

            # kl_loss = - 0.5 * K.sum(1 + z_log_var - z_mean_sq - z_log_var_exp, axis=-1)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            # kl_loss = K.print_tensor(K.mean(kl_loss), message="KL loss is: ")
       
            # return (xent_loss + kl_loss)
            return K.mean(xent_loss + kl_loss)
            # return K.mean(xent_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            # We won't actually use the output.
            return x_decoded_mean

    y = CustomVariationalLayer(name="variational_layer")([x, x_decoded_mean])
    vae = Model(x, y)
    optimizer = Adam()
    vae.compile(optimizer="adam", loss=None)
    vae.summary()
 
    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    encoder_var = Model(x, z_log_var)

    # # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    h = layer_list[0](decoder_input)
    for layer in layer_list[1:]:
        h = layer(h)
    x_decoded_mean = h

    generator = Model(decoder_input, x_decoded_mean)

    return vae, encoder, generator, encoder_var

