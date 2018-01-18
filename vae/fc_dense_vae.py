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
# import data_access as da

from keras.optimizers import Adam
from keras.layers import BatchNormalization, PReLU

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import link_distribution_16 as ld16
# ld = ld16.LinkDistribution()

def sampling(args):
    z_mean, z_log_var = args
    shape_value = (K.shape(z_mean)[0], latent_dim)
    # shape_value = K.print_tensor(shape_value, message="shape_value is: ")
    epsilon = K.random_normal(shape=shape_value, mean=0.,
                              stddev=epsilon_std)

    # z_mean = K.print_tensor(z_mean, message="z_mean is: ")
    # z_log_var = K.print_tensor(z_log_var, message="z_log_var is: ")

    z_sample = z_mean + K.exp(z_log_var) * epsilon/2
    # z_sample = K.print_tensor(z_sample, message="z_sample is: ")
    return z_sample

    # h = Dense(intermediate_dim, activation='relu', name="encoder_1")(x)
    # h = Dense(intermediate_dim, activation='relu', name="encoder_21")(h)
    # h = Dense(intermediate_dim, activation='relu', name="encoder_22")(h)
    # h = Dense(intermediate_dim, activation='relu', name="encoder_23")(h)
    # h = Dense(intermediate_dim, activation='relu', name="encoder_24")(h)
    # h = Dense(intermediate_dim, activation='relu', name="encoder_25")(h)
    # h = Dense(intermediate_dim, activation='relu', name="encoder_26")(h)
    # h = Dense(intermediate_dim, activation='relu', name="encoder_27")(h)
    # # h = Dense(intermediate_dim, activation='relu', name="encoder_28")(h)
    # # h = Dense(intermediate_dim, activation='relu', name="encoder_29")(h)
    # # h = Dense(intermediate_dim, activation='relu', name="encoder_210")(h)

    # decoder_h = Dense(intermediate_dim, activation='relu', name="decoder_1")
    # decoder_h1 = Dense(intermediate_dim, activation='relu', name="decoder_21")
    # decoder_h2 = Dense(intermediate_dim, activation='relu', name="decoder_22")
    # decoder_h3 = Dense(intermediate_dim, activation='relu', name="decoder_23")
    # decoder_h4 = Dense(intermediate_dim, activation='relu', name="decoder_24")
    # decoder_h5 = Dense(intermediate_dim, activation='relu', name="decoder_25")
    # decoder_h6 = Dense(intermediate_dim, activation='relu', name="decoder_26")
    # decoder_h7 = Dense(intermediate_dim, activation='relu', name="decoder_27")
    # # decoder_h8 = Dense(intermediate_dim, activation='relu', name="decoder_28")
    # # decoder_h9 = Dense(intermediate_dim, activation='relu', name="decoder_29")
    # # decoder_h10 = Dense(intermediate_dim, activation='relu', name="decoder_210")

def encoder_network(original_dim, intermediate_dim, latent_dim):
    x = Input(shape=(original_dim,))
    h = BatchNormalization()(x)
    h = PReLU()(h)
    h = Dense(intermediate_dim, name="encoder_1")(x)
    h = BatchNormalization()(h)
    h = PReLU()(h)
    h = Dense(intermediate_dim, name="encoder_21")(h)
    h = BatchNormalization()(h)
    h = PReLU()(h)
    h = Dense(intermediate_dim, name="encoder_22")(h)
    h = BatchNormalization()(h)
    h = PReLU()(h)
    h = Dense(intermediate_dim, name="encoder_23")(h)
    h = BatchNormalization()(h)
    h = PReLU()(h)
    h = Dense(intermediate_dim, name="encoder_24")(h)
    h = BatchNormalization()(h)
    h = PReLU()(h)
    h = Dense(intermediate_dim, name="encoder_25")(h)
    h = BatchNormalization()(h)
    h = PReLU()(h)
    h = Dense(intermediate_dim, name="encoder_26")(h)
    h = BatchNormalization()(h)
    h = PReLU()(h)
    h = Dense(intermediate_dim, name="encoder_27")(h)
    h = BatchNormalization()(h)
    h = PReLU()(h)
    # h = Dense(intermediate_dim, name="encoder_28")(h)
    # h = Dense(intermediate_dim, name="encoder_29")(h)
    # h = Dense(intermediate_dim, name="encoder_210")(h)
    z_mean = Dense(latent_dim, activation='linear', name="encoder_3")(h)
    z_log_var = Dense(latent_dim, activation='linear', name="encoder_4")(h)


    return x, z_mean, z_log_var

def decoder_network(z, intermediate_dim, original_dim):

    layer_list = []
    for i in range(8):
        layer_list.append(Dense(intermediate_dim, name="decoder_"+str(i)))
        layer_list.append(BatchNormalization())
        layer_list.append(PReLU())
    layer_list.append(Dense(original_dim, name="decoder_mean"))
    
    # # we instantiate these layers separately so as to reuse them later
    # decoder_h = Dense(intermediate_dim, name="decoder_1")
    # decoder_h1 = Dense(intermediate_dim, name="decoder_21")
    # decoder_h2 = Dense(intermediate_dim, name="decoder_22")
    # decoder_h3 = Dense(intermediate_dim, name="decoder_23")
    # decoder_h4 = Dense(intermediate_dim, name="decoder_24")
    # decoder_h5 = Dense(intermediate_dim, name="decoder_25")
    # decoder_h6 = Dense(intermediate_dim, name="decoder_26")
    # decoder_h7 = Dense(intermediate_dim, name="decoder_27")
    # # decoder_h8 = Dense(intermediate_dim, activation='relu', name="decoder_28")
    # # decoder_h9 = Dense(intermediate_dim, activation='relu', name="decoder_29")
    # # decoder_h10 = Dense(intermediate_dim, activation='relu', name="decoder_210")
    # decoder_mean = Dense(original_dim, name="decoder_mean")
    h = layer_list[0](z)
    for layer in layer_list[1:]:
        h = layer(h)
    x_decoded_mean = h

    # h = decoder_h(z)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h1(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h2(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h3(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h4(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h5(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h6(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h7(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # # h_decoded = decoder_h8(h_decoded)
    # # h_decoded = decoder_h9(h_decoded)
    # # h_decoded = decoder_h10(h_decoded)

    # x_decoded_mean = decoder_mean(h)
    return layer_list, x_decoded_mean


def get_networks(original_dim, intermediate_dim, latent_dim):

    x, z_mean, z_log_var = encoder_network(original_dim, intermediate_dim, latent_dim)
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    layer_list, x_decoded_mean = decoder_network(z, intermediate_dim, original_dim)
    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean):
            # xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean) 
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
    # h = decoder_h(decoder_input)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h1(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h2(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h3(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h4(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h5(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h6(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # h = decoder_h7(h)
    # h = BatchNormalization()(h)
    # h = PReLU()(h)
    # # _h_decoded = decoder_h8(_h_decoded)
    # # _h_decoded = decoder_h9(_h_decoded)
    # # _h_decoded = decoder_h10(_h_decoded)

    # _x_decoded_mean = decoder_mean(h)

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


# Check gradients
# Check the intermediate values using print_tensors
# Use small input values and check the output


batch_size = 100
original_dim = 10
latent_dim = 10 
intermediate_dim = 1000  
epochs = 1000
num_repeats = 100#20
epsilon_std = 1.0
N = 1000#400000#100000
N_TR = 990#360000#90000
num_ones = 2

x_all = get_k_ones(N, original_dim, num_ones)

# x_all = da.get_data(True)
# x_all = da.split_vector(x_all)
# print(x_all.shape)

x_all = x_all[:N]
x_train = x_all[:N_TR]
x_test = x_all[N_TR:]



vae, encoder, generator, encoder_var = \
    get_networks(original_dim, intermediate_dim, latent_dim)


num_valid_pred = []
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


    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None),
            # callbacks=[
            #     tensorboard,
            # ]
            )
    vae.save_weights("vae_"+str(num_epocs)+".h5")

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
    


    ld16.validate_sw_distribution(ytp)

    # print("Validation Ends")

    num_validation = 10#10000
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