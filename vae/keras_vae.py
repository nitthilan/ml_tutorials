'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import norm

from keras.layers import Input, Dense, Dropout, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

import pickle
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import link_distribution_16 as ld16
ld = ld16.LinkDistribution()


batch_size = 10
original_dim = 480
latent_dim = 2
intermediate_dim = 256
epochs = 10
epsilon_std = 1.0

# Encoder Network
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
h = Dense(intermediate_dim, activation='relu')(h)
h = Dense(intermediate_dim, activation='relu')(h)
h = Dense(intermediate_dim, activation='relu')(h)
h = Dense(intermediate_dim, activation='relu')(h)

# h = Dropout(0.25)(h1)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    z_log_var = K.print_tensor(z_log_var, message="z_log_var sampled: ")
    z_mean = K.print_tensor(z_mean, message="z_mean sampled: ")
    epsilon = K.print_tensor(epsilon, message="epsilon sampled: ")
    z_output = z_mean + K.exp(z_log_var / 2) * epsilon
    z_output = K.print_tensor(z_output, message="z_output sampled: ")
    return z_output


# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_h1 = Dense(intermediate_dim, activation='relu')
decoder_h2 = Dense(intermediate_dim, activation='relu')
decoder_h3 = Dense(intermediate_dim, activation='relu')
decoder_h4 = Dense(intermediate_dim, activation='relu')

decoder_mean = Dense(original_dim, activation='sigmoid')


# Decoder network
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

h_decoded = decoder_h(z)
h_decoded = decoder_h1(h_decoded)
h_decoded = decoder_h2(h_decoded)
h_decoded = decoder_h3(h_decoded)
h_decoded = decoder_h4(h_decoded)

# h_decoded = Dropout(0.25)(h_decoded)
x_decoded_mean = decoder_mean(h_decoded)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        # xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        xent_loss = metrics.binary_crossentropy(x, x_decoded_mean)        
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # xent_loss = K.print_tensor(xent_loss, message="xent loss is: ")
        # kl_loss = K.print_tensor(kl_loss, message="KL loss is: ")
        # return K.mean(xent_loss + kl_loss)
        return K.mean(xent_loss+kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)



NUM_DATA = 100000
TRAIN_OFFSET = int(0.9*NUM_DATA)

# DUMP_FILE = "design_data_latent.pickle"
DUMP_FILE = "design_100000.npz"
IS_STORED = True

def save_data(x):

    np.savez(DUMP_FILE, x=x)
    
    # Saving the objects:
    # with open(DUMP_FILE, 'w') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump(x, f)

    return
def return_saved():
    return np.load(DUMP_FILE)['x']
    
    # Saving the objects:
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
        return rand_designs[:NUM_DATA]

# train the VAE on MNIST digits
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

data_x = get_data(IS_STORED)
x_train = data_x[:TRAIN_OFFSET,:]
x_test = data_x[TRAIN_OFFSET:,:]


vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)
encoder_var = Model(x, z_log_var)

x_test_encoded_var = encoder_var.predict(x_test, batch_size=batch_size)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.scatter(x_test_encoded_var[:, 0], x_test_encoded_var[:, 1])

# plt.colorbar()
# plt.show()
plt.savefig("../../../../data/test_output/output.jpg")
print(x_test_encoded_var)

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 5  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        # z_sample = np.array([[xi, yi, xi, yi, xi, yi, xi, yi, xi, yi]])
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        # print(x_decoded[0])
        # t = x_decoded[0]
        # t[t<0.25] = 0
        # t[t>=0.25] = 1
        # print(t)
        # print(np.sum(t))
#         digit = x_decoded[0].reshape(digit_size, digit_size)
#         figure[i * digit_size: (i + 1) * digit_size,
#                j * digit_size: (j + 1) * digit_size] = digit

# plt.figure(figsize=(10, 10))
# plt.imshow(figure, cmap='Greys_r')
# plt.show()
