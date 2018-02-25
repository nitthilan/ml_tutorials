from __future__ import print_function

import sys, os
import numpy as np
from keras.datasets import cifar10, cifar100
import keras
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.get_cifar10 as gc

(x_train, y_train), (x_test, y_test) = \
	gc.get_reduced_class_data(10, [2,3])

# print(np.bincount(y_test))
# print(np.bincount(y_train))

def get_conv_net_small(input_shape, num_classes, num_extra_conv_layers):
	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same',
	                 input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	for i in range(num_extra_conv_layers):
		model.add(Conv2D(64, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

	model.add(MaxPooling2D(pool_size=(8, 8)))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	# initiate RMSprop optimizer
	# opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
	opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])

	model.summary()

	return model

model = get_conv_net_small(x_train.shape[1:], 2, 0)

batch_size = 128
epochs = 500
print('Not using data augmentation.')
model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True)