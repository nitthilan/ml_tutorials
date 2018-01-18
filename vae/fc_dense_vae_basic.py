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
import data_load.tsv_data_load as tda


# List of things to try:
# - Use CNN 1D to generate sequences instead of fully connected network
# - The TSV data is discrete integer data so the error
# - Can we just train a generator which maps a random input gaussian input to 
# - RNNs with one hot encoding of number of tsvs and Embedding layer


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
    layer_list.append(Dense(original_dim, activation='softplus', name="decoder_mean"))
    # layer_list.append(Dense(original_dim, activation='sigmoid', name="decoder_mean"))
    
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
            xent_loss =  original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
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
original_dim = 100
latent_dim = 100 
intermediate_dim = 500  
epochs = 10
num_repeats = 1000#20
epsilon_std = 1.0
N = 100000#1000#400000#100000
N_TR = 90000#990#360000#90000
num_ones = 19#2
num_inter_layers = 8
data_base_folder = "../../data/ml_tutorials/vae/sig_bin_cross_1/"
# data_base_folder = "../../data/vae/mse_softplus/"

x_all = get_k_ones(N, original_dim, num_ones)

# x_all = lda.get_data(True)
# x_all = lda.split_vector(x_all)

# x_all = tda.get_data(True, 19, N)

print(x_all.shape)

x_all = x_all[:N]
x_train = x_all[:N_TR]
x_test = x_all[N_TR:]



vae, encoder, generator, encoder_var = \
    get_networks(original_dim, \
        intermediate_dim, latent_dim, num_inter_layers)


num_valid_pred = []
# vae.load_weights(data_base_folder+"/vae_0.h5")
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
    vae.save_weights(data_base_folder+"/vae_"+str(num_epocs)+".h5")

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
    # mean_layer = K.function([vae.layers[0].input, K.learning_phase()],
    #                         [vae.layers[25].output])
    # var_layer = K.function([vae.layers[0].input, K.learning_phase()],
    #                         [vae.layers[26].output])

    # mean_output = mean_layer([x_train, 0])[0]
    # variance_output = var_layer([x_train, 0])[0]

    # print(mean_output.shape, variance_output.shape)
    # print(np.sum(np.abs(mean_output), axis=1)[:10], np.sum(np.abs(variance_output), axis=1)[:10])

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
    # x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    # fig = plt.figure(figsize=(6, 6))
    # plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
    # # plt.colorbar()
    # # plt.show()
    # fig.savefig("vae_encoded.png")

    # ytp = vae.predict(x_train, batch_size=batch_size)
    # ytp[ytp<0.5] = 0
    # ytp[ytp>=0.5] = 1
    # ytp = np.concatenate((ytp, ytp, ytp, ytp), axis=1)
    # validate_sw_distribution(ytp)

    # print("Validation Ends")

    num_validation = 1000
    z_sample = np.random.normal(loc=0.0, scale=epsilon_std, \
        size=(num_validation,latent_dim))
    x_decoded = generator.predict(z_sample)
    print(x_decoded[:10])
    x_dec_ones = np.round(np.copy(x_decoded))
    print(x_dec_ones[:10])
    x_dec_sum = np.sum(x_dec_ones, axis=1)
    num_equal = np.sum((x_dec_sum == num_ones))
    print("Num valid vs iterations ", num_epocs, num_equal, num_validation)
    num_valid_pred.append(num_equal)


print("Num Valid across iterations ", num_validation)
print(num_valid_pred)
