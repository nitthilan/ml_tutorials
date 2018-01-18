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

import gen_fc_net as gfn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import debug_utils.debug_utils as du


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

N = 10000#0#1000#400000#100000
N_TR = 9000#0#990#360000#90000
num_ones = 2#2

original_dim = 100
latent_dim = 10 
intermediate_dim = 1000
num_inter_layers = 8


batch_size = 100
epochs = 10
num_repeats = 1000#20

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
    gfn.get_networks(original_dim, \
        intermediate_dim, latent_dim, num_inter_layers)

du.dump_layer_details(vae)

num_valid_pred = []
# vae.load_weights(data_base_folder+"/vae_0.h5")
# print("Loaded weights New")
for num_epocs in range(num_repeats):

    du.dump_weights(vae)
    
    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None),
            )
    vae.save_weights(data_base_folder+"/vae_"+str(num_epocs)+".h5")

    # du.dump_gradients(vae, x_train)

    mean_output = du.get_intermediate_output(vae, 0, 25, x_train)
    variance_output = du.get_intermediate_output(vae, 0, 26, x_train)

    print(mean_output.shape, variance_output.shape)
    print(np.mean(np.abs(mean_output), axis=0), np.exp(np.mean(variance_output, axis=0)))

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
    z_sample = np.random.normal(loc=0.0, scale=1.0, \
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
