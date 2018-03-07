
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers

def get_conv_net(x_shape, num_classes, num_extra_conv_layers,
    wgt_fname=None):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    if(num_extra_conv_layers == 2):
        skip_layer_a = False
        skip_layer_b = False
    elif(num_extra_conv_layers == 1):
        skip_layer_a = False
        skip_layer_b = True
    else:
        skip_layer_a = True
        skip_layer_b = True

    model = Sequential()
    weight_decay = 0.0005

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

#     if(skip_layer_a == False):
# Epoch 198/198
# 390/390 [==============================] - 8s 22ms/step - loss: 0.2291 - acc: 0.9843 - val_loss: 0.5393 - val_acc: 0.9106
# Epoch 199/199
# 390/390 [==============================] - 8s 22ms/step - loss: 0.2300 - acc: 0.9843 - val_loss: 0.5393 - val_acc: 0.9104

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    if(skip_layer_a == False):

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


    if(skip_layer_b == False):
        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # model.add(Dropout(0.5))


    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    top_model.add(Activation('relu'))
    top_model.add(BatchNormalization())

    top_model.add(Dense(2048,kernel_regularizer=regularizers.l2(weight_decay)))
    top_model.add(Activation('relu'))
    top_model.add(BatchNormalization())

    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes))
    top_model.add(Activation('softmax'))


    # If a already trained model is present
    # copy weights from one model to another
    if(wgt_fname):
        model.load_weights(wgt_fname, by_name=True)
        
        for layer in model.layers:
            layer.trainable = False

        # top_model.load_weights(wgt_fname, by_name=True)

    model.add(top_model)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.summary()
    return model


def get_conv_net_v1(x_shape, num_classes, 
    num_extra_conv_layers, wgt_fname=None):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    model = Sequential()
    weight_decay = 0.0005

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # -------- Net 4 ------------------------
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # -------- Net 4 ------------------------

    if(num_extra_conv_layers == 2 or 
       num_extra_conv_layers == 1):

        model.add(MaxPooling2D(pool_size=(2, 2)))

    #     if(skip_layer_a == False):
    # Epoch 198/198
    # 390/390 [==============================] - 8s 22ms/step - loss: 0.2291 - acc: 0.9843 - val_loss: 0.5393 - val_acc: 0.9106
    # Epoch 199/199
    # 390/390 [==============================] - 8s 22ms/step - loss: 0.2300 - acc: 0.9843 - val_loss: 0.5393 - val_acc: 0.9104

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

    if(num_extra_conv_layers == 2):

        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.4))

        # model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.4))

        # model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization())
    
    # model.add(Dropout(0.5))


    top_model = Sequential()
    
    if(num_extra_conv_layers == 2):
        top_model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(4, 4)))
    elif(num_extra_conv_layers == 1):
        top_model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(4, 4)))
    else:
        top_model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(8, 8)))

    top_model.add(Flatten())
    top_model.add(Dense(256,kernel_regularizer=regularizers.l2(weight_decay)))
    top_model.add(Activation('relu'))
    top_model.add(BatchNormalization())

    # top_model.add(Dense(2048,kernel_regularizer=regularizers.l2(weight_decay)))
    # top_model.add(Activation('relu'))
    # top_model.add(BatchNormalization())

    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes))
    top_model.add(Activation('softmax'))


    # If a already trained model is present
    # copy weights from one model to another
    if(wgt_fname):
        model.load_weights(wgt_fname, by_name=True)
        
        for layer in model.layers:
            layer.trainable = False

        # top_model.load_weights(wgt_fname, by_name=True)

    model.add(top_model)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.summary()
    return model


def transfer_weights(old_model, new_model):
    return