
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
from keras.constraints import unit_norm

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




def get_conv_vert_net(x_shape, num_classes, num_vert_filters,
    use_bias=True):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    model = Sequential()
    weight_decay = 0.0005

    # if(num_vert_filters == 4):
    #     F1 = 64; F2 =128; F3 =256; F4 = 512; D1 = 512; D2 = 128;
    # elif(num_vert_filters == 3):
    #     F1 = 48; F2 = 96; F3 = 192; F4 = 384; D1 = 384; D2 = 112;
    # elif(num_vert_filters == 2):
    #     F1 = 40; F2 = 80; F3 = 160; F4 = 320; D1 = 320; D2 = 96;
    #     # F1 = 32; F2 = 64; F3 = 128; F4 = 256; D1 = 256; D2 = 64;
    # else: # num_vert_filters == 1
    #     F1 = 16; F2 = 32; F3 = 64; F4 = 128; D1 = 128; D2 = 32;

    # temp_num_filter = num_vert_filters
    # num_vert_filters = 4
    F1 = int(16*num_vert_filters); F2 = int(32*num_vert_filters); 
    F3 = int(64*num_vert_filters); F4 = int(128*num_vert_filters); 
    D1 = int(128*num_vert_filters); D2 = int(32*num_vert_filters);

    # F1 = int(16*num_vert_filters); F2 = int(32*num_vert_filters); 
    # F3 = int(32*num_vert_filters); F4 = int(32*num_vert_filters); 
    # D1 = int(128*num_vert_filters); D2 = int(32*num_vert_filters);

    model.add(Conv2D(F1, (3, 3), use_bias=use_bias, padding='same', 
        input_shape=x_shape)) 
        # kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    model.add(Conv2D(F1, (3, 3), use_bias=use_bias, padding='same'
        , kernel_constraint=unit_norm(axis=[0, 1, 2])))
        # , kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))


    # F2 = int(32*temp_num_filter); 
    # F3 = int(64*temp_num_filter); F4 = int(128*temp_num_filter); 
    # D1 = int(128*temp_num_filter); D2 = int(32*temp_num_filter);


    model.add(Conv2D(F2, (3, 3), padding='same',use_bias=use_bias
        , kernel_constraint=unit_norm(axis=[0, 1, 2])))
        # , kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(F2, (3, 3), padding='same',use_bias=use_bias
        , kernel_constraint=unit_norm(axis=[0, 1, 2])))
        # , kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(F3, (3, 3), padding='same',use_bias=use_bias
        , kernel_constraint=unit_norm(axis=[0, 1, 2])))
        # , kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(F3, (3, 3), padding='same',use_bias=use_bias
        , kernel_constraint=unit_norm(axis=[0, 1, 2])))
        # , kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(F3, (3, 3), padding='same',use_bias=use_bias
        , kernel_constraint=unit_norm(axis=[0, 1, 2])))
        # , kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(F4, (3, 3), padding='same',use_bias=use_bias
        , kernel_constraint=unit_norm(axis=[0, 1, 2])))
        # , kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(F4, (3, 3), padding='same',use_bias=use_bias
        , kernel_constraint=unit_norm(axis=[0, 1, 2])))
        # , kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(F4, (3, 3), padding='same',use_bias=use_bias
        , kernel_constraint=unit_norm(axis=[0, 1, 2])))
        # , kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(4, 4)))


    # model.add(Conv2D(F4, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    # model.add(Conv2D(F4, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    # model.add(Conv2D(F4, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())

    # model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # model.add(Dropout(0.5))

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    # # if(num_vert_filters == 1):
    # top_model.add(Dense(D1))#,kernel_regularizer=regularizers.l2(weight_decay)))
    # top_model.add(Activation('relu'))
    # top_model.add(BatchNormalization())

    top_model.add(Dense(D2))#, kernel_constraint=unit_norm()))
        # ,use_bias=True, kernel_regularizer=regularizers.l2(weight_decay)))
    top_model.add(Activation('relu'))
    # top_model.add(BatchNormalization())

    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes,use_bias=True))#, kernel_constraint=unit_norm()))
    top_model.add(Activation('softmax'))

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

    if(num_extra_conv_layers == 3 or
       num_extra_conv_layers == 2 or 
       num_extra_conv_layers == 1):

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    # -------- Net 4 ------------------------

    if(num_extra_conv_layers == 3 or 
       num_extra_conv_layers == 2):

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

    if(num_extra_conv_layers == 3):

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
    
    model.add(Dropout(0.5))


    top_model = Sequential()
    
    if(num_extra_conv_layers == 3):
        top_model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(2, 2)))
    elif(num_extra_conv_layers == 2):
        top_model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(4, 4)))
        # top_model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(8, 8)))
    elif(num_extra_conv_layers == 1):
        top_model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(8, 8)))
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

def get_conv_net_cifar100(x_shape, num_classes, 
    num_extra_conv_layers, wgt_fname=None):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    model = Sequential()
    weight_decay = 0.0005

    model.add(Conv2D(128, (3, 3), padding='same',
                     input_shape=x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # -------- Net 4 ------------------------
    if(num_extra_conv_layers == 3 or
       num_extra_conv_layers == 2 or 
       num_extra_conv_layers == 1):

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    # -------- Net 4 ------------------------
    if(num_extra_conv_layers == 3 or 
       num_extra_conv_layers == 2):

        model.add(MaxPooling2D(pool_size=(2, 2)))

    #     if(skip_layer_a == False):
    # Epoch 198/198
    # 390/390 [==============================] - 8s 22ms/step - loss: 0.2291 - acc: 0.9843 - val_loss: 0.5393 - val_acc: 0.9106
    # Epoch 199/199
    # 390/390 [==============================] - 8s 22ms/step - loss: 0.2300 - acc: 0.9843 - val_loss: 0.5393 - val_acc: 0.9104

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

    if(num_extra_conv_layers == 3):

        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(1024, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(1024, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(1024, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(1024, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(1024, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(1024, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    
    model.add(Dropout(0.5))


    top_model = Sequential()
    
    if(num_extra_conv_layers == 3):
        top_model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(2, 2)))
    elif(num_extra_conv_layers == 2):
        top_model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(4, 4)))
        # top_model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(8, 8)))
    elif(num_extra_conv_layers == 1):
        top_model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(8, 8)))
    else:
        top_model.add(MaxPooling2D(input_shape=model.output_shape[1:],pool_size=(8, 8)))        

    top_model.add(Flatten())

    # if(num_extra_conv_layers == 3):
    #     top_model.add(Dense(1024,kernel_regularizer=regularizers.l2(weight_decay)))
    #     top_model.add(Activation('relu'))
    #     top_model.add(BatchNormalization())

    top_model.add(Dropout(0.5))
    top_model.add(Dense(1024,kernel_regularizer=regularizers.l2(weight_decay)))
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
