'''This script demonstrates how to build a variational autoencoder with Keras.
 #Reference
 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
import matplotlib as mpl
mpl.use('Agg')
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.ld_data_load as lda

# List of Queries:
# - Tutorial on Variational AutoEncoder
# Why is the encoder required for VAE? 
#   - Why cannot the input be a random generated Gaussian noise?
# Understand how the cost function and the sampling works?
# Implement a RNN based VAE?
#   - https://arxiv.org/pdf/1506.02216.pdf
#       - A Recurrent Latent Variable Model for Sequential Data
# Why is the sampling values different in different models?
#   - https://blog.keras.io/building-autoencoders-in-keras.html
#   - This differes from what I have implemented and refered


# Tutorial on Variational Autoencoders: https://arxiv.org/pdf/1606.05908.pdf
# List of things to learn:
# List of Hyper parameters:
# How to estimate the step size
#   - What should be the first value?
#   - How should you decrease it?
# Weight initialisation
#   - How should you initialise the weights
# What kind of regularizer should one use?
#   - L2/L1
# Minibatch size:
#   - 32/64/128
# binary cross entropy and softmax

# Use Batch normalisation and Proper weight initialisation    

# Things to try:
#   - RNN based VAE: 
#   - Have a Encoder which encodes the state into state and decoder which generates the output
#   - Add a Posterior Regularization factor for the condition to be enforced


# Autoencoder
# https://blog.keras.io/building-autoencoders-in-keras.html
# https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

# Debugging weights and updates:
# https://keras.io/callbacks/ - Tensorflow dumps
# http://playground.tensorflow.org/ - checking the playground

# Debugging tool available in Keras:
# history object retuned by model to plot accuracy and 
# Tensorflow:

# GAN Learning Tutorials:
# https://arxiv.org/pdf/1708.01729.pdf - Inception Score, Label Smoothing, 
# https://github.com/aleju/papers/blob/master/neural-nets/Improved_Techniques_for_Training_GANs.md
# 

# https://www.youtube.com/watch?v=Xogn6veSyxA - Ch 9: Convolutional Networks
#   - Ian GoodFellow: Batch Normalization and Convolutional Networks
# https://www.youtube.com/watch?v=ckoD_bE8Bhs - IanGoodfellow PhD Defense Presentation
#   - 

# ########################### From old files

# Why not use just 1D for latent variable?
# How does 3D work? Check the error for higher dimensions?
# http://kvfrans.com/generative-adversial-networks-explained/

# https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
# https://arxiv.org/pdf/1606.05908.pdf
# https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/

# Links to refer:
# https://github.com/bstriner/keras-adversarial
# https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
# http://lantaoyu.com/files/2017-07-26-gan-for-discrete-data.pdf
# https://keras.io/backend/
# https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0

# https://github.com/mkusner/grammarVAE - discrete vae

# VAE:
# Posterior Regularization: http://www.jmlr.org/papers/volume11/ganchev10a/ganchev10a.pdf
# Stein Discrepancy (SD) is a good concept to know about. It has better properties than KL divergence between distributions. Posterior regularization framework employs KL distance. So if it is appropriate, we can consider replacing it with SD.

# http://www.cs.dartmouth.edu/~qliu/PDF/ksd_short.pdf
# https://arxiv.org/pdf/1602.03253v2.pdf
# http://www.cs.dartmouth.edu/~qliu/PDF/slides_ksd_icml2015.pdf
# https://arxiv.org/pdf/1602.02964.pdf
# http://arxiv.org/abs/1608.04471

# References for Variational Autoencoders:
# - https://jaan.io/what-is-variational-autoencoder-vae-tutorial/ 
# - http://dustintran.com/blog/variational-auto-encoders-do-not-train-complex-generative-models 
# - https://arxiv.org/pdf/1606.05908.pdf
# - http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf 
# - https://arxiv.org/pdf/1312.6114.pdf
# - https://arxiv.org/pdf/1606.04934.pdf
# - https://arxiv.org/pdf/1706.04223.pdf

# Tutorial for Posterior Regularization:
# - https://homes.cs.washington.edu/~taskar/pubs/druck_ganchev_graca_11.pdf
# - http://www.jmlr.org/papers/volume11/ganchev10a/ganchev10a.pdf (Beautiful paper from late Ben Taskar and his students)

# http://timvieira.github.io/blog/post/2014/10/06/kl-divergence-as-an-objective-function/ 
# This will be useful in the context of Posterior Regularization.

# GAN:
# http://www.rricard.me/machine/learning/generative/adversarial/networks/keras/tensorflow/2017/04/05/gans-part2.html
# https://github.com/osh/KerasGAN
# https://github.com/bstriner/keras-adversarial
# NIPS tutorial GAN: https://arxiv.org/pdf/1701.00160.pdf
# http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf


def sampling(args):
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
    # layer_list.append(Dense(original_dim, activation='softplus', name="decoder_mean"))
    layer_list.append(Dense(original_dim, activation='sigmoid', name="decoder_mean"))
    
    # we instantiate these layers separately so as to reuse them later
    h = layer_list[0](z)
    for layer in layer_list[1:]:
        h = layer(h)
    x_decoded_mean = h

    return layer_list, x_decoded_mean


def get_networks(original_dim, intermediate_dim, latent_dim, num_layers):

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
            xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean) 
            # xent_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean) 
            # xent_loss = K.print_tensor(K.mean(xent_loss),  message="xent loss is: ")

            z_log_var_exp = K.exp(z_log_var)
            z_mean_sq = K.square(z_mean)
            # z_log_var_exp = K.print_tensor(z_log_var_exp, message="z_log_var avg: ")
            # z_mean_sq = K.print_tensor(z_mean_sq,  message="z_mean avg: ")

            # kl_loss = - 0.5 * K.sum(1 + z_log_var - z_mean_sq - z_log_var_exp, axis=-1)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            # kl_loss = K.print_tensor(K.mean(kl_loss), message="KL loss is: ")
       
            # return (xent_loss + kl_loss)
            # return K.mean(xent_loss + kl_loss)
            return K.mean(xent_loss)

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


# # train the VAE on MNIST digits
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
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

X_WTH = 4
Y_HGT = 4
Z_LYR = 4
def GET_NUM_CONNECTION():
    num_cores_in_plane = X_WTH*Y_HGT
    num_connection_in_plane = (num_cores_in_plane*(num_cores_in_plane-1))/2
    return int(Z_LYR*num_connection_in_plane)

def get_node_index(position):
    z,y,x = position
    return int(z*X_WTH*Y_HGT + y*X_WTH + x)

def get_2d_core_pos():
    # Since all the connection is in a plane we can generate the 
    # connection information for a single plane and replicate it for all planes

    # Generate all the (x,y) positions
    core_pos = []
    for i in range(Y_HGT):
        for j in range(X_WTH):
            core_pos.append((i,j))
    return core_pos
def get_distance(x,y,x1,y1):
    return (((x-x1)**2+(y-y1)**2)**0.5)

def get_core_distance(start,end):
    return get_distance(start[0], start[1], end[0], end[1])


# Generate all the possible connection options that are possible
# The order in which they are generated is used for generating feature vectors
def generate_core_connection_options():
    core_pos = get_2d_core_pos()
    total_cores = len(core_pos)
    core_connection_options = {}
    for i in range(4):
        core_connection_options[i] = []
    #
    for i in range(total_cores):
        for j in range(i+1,total_cores):
            distance = int(round(get_core_distance(core_pos[i], core_pos[j])))
            start_core = get_node_index((0, core_pos[i][1], core_pos[i][0]))
            end_core = get_node_index((0, core_pos[j][1], core_pos[j][0]))
            # Store the distance, start and end core idx
            # core_connection_options.append((start_core, end_core, distance))
            core_connection_options[distance-1].append((start_core, end_core))
            
    # print("Length of connection option ", len(core_connection_options))
    # for i in range(4):
    #   print(len(core_connection_options[i]))

    #
    core_connection_ordering = []
    for i in range(4):
        for conn in core_connection_options[i]:
            (start_core, end_core) = conn
            core_connection_ordering.append((start_core, end_core, i))

    # (42, 40, 28, 10)
    # print(len(core_connection_options[0]), len(core_connection_options[1]),
    #   len(core_connection_options[2]), len(core_connection_options[3]))
    # 
    # prob_ord_idx_list = []
    # con_idx = 0
    # for i in range(4):
    #   for j in range(len(core_connection_options[i])):
    #       for k in range(C.DISTRIBUTION[i]):
    #           prob_ord_idx_list.append(con_idx)
    #       con_idx += 1

    # print(prob_ord_idx_list)



    return core_connection_options, core_connection_ordering

def generate_conn_idx_list_list(feature_vector_list):
    connection_idx_list_list = []
    (N, M) = feature_vector_list.shape
    # print("Shape ", N, M)

    for i in range(N):
        connection_idx_list = []
        for j in range(GET_NUM_CONNECTION()):
            if(feature_vector_list[i][j]):
                connection_idx_list.append(j)
        connection_idx_list_list.append(connection_idx_list)

    return connection_idx_list_list

def validate_sw_distribution(feature_vector):
    _, core_connection_ordering = generate_core_connection_options()
    connection_list_list = generate_conn_idx_list_list(feature_vector)
    ref_dist = [64, 20, 8, 4]
    i = 0
    for connection_list in connection_list_list:
        dist_hist = []
        for idx in range(4):
            dist_hist.append(0)
        for conn in connection_list:
            conn = conn%120
            x,y,dist = core_connection_ordering[conn]
            dist_hist[dist] += 1

        if(dist_hist != ref_dist):
            print("Error ",i, dist_hist)
        # print(dist_hist)
        i+=1
    print("Done validation all checks")
    return


# Check gradients
# Check the intermediate values using print_tensors
# Use small input values and check the output


batch_size = 100
original_dim = 120
latent_dim = 10 
intermediate_dim = 500  
epochs = 10
num_repeats = 1000#20
epsilon_std = 1.0
N = 400000#1000#400000#100000
N_TR = 360000#990#360000#90000
num_ones = 2
num_inter_layers = 10
# data_base_folder = "../../data/vae/sigmoid_bin_cross"
data_base_folder = "../../data/vae/mse_softplus"

# x_all = get_k_ones(N, original_dim, num_ones)

x_all = lda.get_data(True)
x_all = lda.split_vector(x_all)
print(x_all.shape)

x_all = x_all[:N]
x_train = x_all[:N_TR]
x_test = x_all[N_TR:]



vae, encoder, generator, encoder_var = \
    get_networks(original_dim, \
        intermediate_dim, latent_dim, num_inter_layers)


num_valid_pred = []
# vae.load_weights("../../data/vae/weights.50-10.55.hdf5")
# print("Loaded weights New")
for num_epocs in range(num_repeats):

    # tensorboard = TensorBoard(log_dir='./logs/vae_'+str(num_epocs), 
    #     histogram_freq=1, batch_size=32, 
    #     write_graph=False, write_grads=True, write_images=False, 
    #     embeddings_freq=0, embeddings_layer_names=None, 
    #     embeddings_metadata=None)

    # Dump the mean weight and std deviation
    layer_idx = 0
    for layer in vae.layers:
        if(len(layer.weights)==2):
            weights = layer.get_weights()
            weight_vector = weights[0]
            bias_vector = weights[1]
            # print("weights ",layer.name, \
            #     weight_vector.shape, bias_vector.shape)
            print(layer.name, "wgt ", np.mean(weight_vector), \
                np.std(weight_vector),\
                "bias ", np.mean(bias_vector), np.std(bias_vector),)
        layer_idx += 1

    modelCheckpoint = ModelCheckpoint(data_base_folder+"weights.{epoch:02d}-{val_loss:.2f}.hdf5", 
        monitor='loss', verbose=0, save_best_only=True, 
        save_weights_only=True, mode='auto', period=1)


    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None),
            callbacks=[
                # modelCheckpoint
                #   earlyStopping, 
                #   reduceonplateau,
                #   csv_logger
                ]
            )
    vae.save_weights("../../data/vae/sigmoid_bin_cross/vae_"+str(num_epocs)+".h5")

    # with a Sequential model
    layer_idx = 0
    for layer in vae.layers:
        # print(layer_idx, layer.name, \
        #     layer.input_shape, layer.output_shape)
        # if(len(layer.weights)):
        #     print("weights ",layer.get_weights())
        # print("updates ",layer.get_updates_for(0))
        layer_idx += 1
    # print(dir(vae.layers[0]))
    mean_layer = K.function([vae.layers[0].input, K.learning_phase()],
                            [vae.layers[25].output])
    var_layer = K.function([vae.layers[0].input, K.learning_phase()],
                            [vae.layers[26].output])

    mean_output = mean_layer([x_train, 0])[0]
    variance_output = var_layer([x_train, 0])[0]

    # print(mean_output.shape, variance_output.shape)
    print(np.sum(np.abs(mean_output), axis=1)[:10], np.sum(np.abs(variance_output), axis=1)[:10])

    # https://github.com/keras-team/keras/issues/2226 - How to get gradients
    weights = vae.trainable_weights # weight tensors
    # weights = [layer.get_weights() for layer in vae.layers if layer.trainable] # filter down weights tensors to only ones which are trainable
    gradients = vae.optimizer.get_gradients(vae.total_loss, weights) # gradient tensors

    # print(vae.total_loss, len(weights))
    # print(gradients)

    # input_tensors = [vae.inputs[0], # input data
    #              vae.sample_weights[0], # how much to weight each sample by
    #              vae.targets[0], # labels
    #              K.learning_phase(), # train or test mode
    #             ]

    # get_gradients = K.function(inputs=input_tensors, outputs=gradients)

    # # inputs = [x_test, # X
    # #       [1], # sample weights
    # #       x_test, # y
    # #       0 # learning phase in TEST mode
    # # ]

    # for (weight_name, gradients) in zip(weights, get_gradients([x_train])):
    #     print(weight_name.name, np.mean(gradients), np.std(gradients))


    # display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
    # plt.colorbar()
    # plt.show()
    fig.savefig("vae_encoded.png")

    ytp = vae.predict(x_train, batch_size=batch_size)
    ytp[ytp<0.5] = 0
    ytp[ytp>=0.5] = 1
    ytp = np.concatenate((ytp, ytp, ytp, ytp), axis=1)
    


    validate_sw_distribution(ytp)

    # print("Validation Ends")

    num_validation = 10000
    z_sample = np.random.normal(loc=0.0, scale=epsilon_std, \
        size=(num_validation,latent_dim))
    x_decoded = generator.predict(z_sample)
    print(x_decoded[:10])
    x_dec_ones = np.copy(x_decoded)
    x_dec_ones[x_decoded<0.5] = 0
    x_dec_ones[x_decoded>=0.5] = 1
    print(x_dec_ones[:10])
    x_dec_sum = np.sum(x_dec_ones, axis=1)
    num_equal = np.sum((x_dec_sum == num_ones))
    print("Num valid vs iterations ", num_epocs, num_equal, num_validation)
    num_valid_pred.append(num_equal)


print("Num Valid across iterations ", num_validation)
print(num_valid_pred)
# # display a 2D manifold of the digits
# n = 15  # figure with 15x15 digits
# digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))
# # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# # to produce values of the latent variables z, since the prior of the latent space is Gaussian
# grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample = np.array([[xi, yi]])
#         x_decoded = generator.predict(z_sample)
#         digit = x_decoded[0].reshape(digit_size, digit_size)
#         figure[i * digit_size: (i + 1) * digit_size,
#                j * digit_size: (j + 1) * digit_size] = digit

# fig = plt.figure(figsize=(10, 10))
# plt.imshow(figure, cmap='Greys_r')
# # plt.show()
# fig.savefig("vae_output.png")