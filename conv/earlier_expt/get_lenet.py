from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.regularizers import l2
from keras import regularizers
from keras import optimizers


def get_conv_net_v0(input_shape, classes, weightsPath=None):
        # initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL
        model.add(Conv2D(20, 5, 5, border_mode="same",
            input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Conv2D(50, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        model.summary()

        # if weightsPath is specified load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model

# 
def get_conv_net(input_shape, classes, 
	weightsPath=None):
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', 
    	kernel_initializer='he_normal', input_shape=input_shape,
    	kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', 
    	kernel_initializer='he_normal',
    	kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(classes, activation = 'softmax', kernel_initializer='he_normal'))
    # sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model

def get_conv_net_conf(input_shape, classes, 
	nf_conv1, nf_conv2, nf_dense1, nf_dense2,
	is_reg1, is_reg2, lr, momentum):
	model = Sequential()
	if(is_reg1):
	    model.add(Conv2D(nf_conv1, (5, 5), 
	    	padding='valid', activation = 'relu', 
	    	kernel_initializer='he_normal', input_shape=input_shape,
	    	kernel_regularizer=regularizers.l2(0.01)))
	else:
		model.add(Conv2D(nf_conv1, (5, 5), 
	    	padding='valid', activation = 'relu', 
	    	kernel_initializer='he_normal', input_shape=input_shape))

	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	if(is_reg2):
	    model.add(Conv2D(nf_conv2, (5, 5), 
	    	padding='valid', activation = 'relu', 
	    	kernel_initializer='he_normal',
	    	kernel_regularizer=regularizers.l2(0.01)))
	else:
		model.add(Conv2D(nf_conv2, (5, 5), 
	    	padding='valid', activation = 'relu', 
	    	kernel_initializer='he_normal'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Flatten())
	model.add(Dense(nf_dense1, activation = 'relu', kernel_initializer='he_normal'))
	model.add(Dense(nf_dense2, activation = 'relu', kernel_initializer='he_normal'))
	model.add(Dense(classes, activation = 'softmax', kernel_initializer='he_normal'))
	model.summary()
	sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	return model