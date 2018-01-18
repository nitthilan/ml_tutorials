List of things to try:
- Use CNN 1D to generate sequences instead of fully connected network
- The TSV data is discrete integer data so the error
- Can we just train a generator which maps a random input gaussian input to 
- RNNs with one hot encoding of number of tsvs and Embedding layer

# Check gradients
# Check the intermediate values using print_tensors
# Use small input values and check the output



References:


OLD:
# http://szhao.me/2017/06/10/a-tutorial-on-mmd-variational-autoencoders.html
# https://github.com/cdoersch/vae_tutorial
# https://arxiv.org/pdf/1606.05908.pdf
# 
# Why not use just 1D for latent variable?
# How does 3D work? Check the error for higher dimensions?


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


