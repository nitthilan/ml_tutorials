from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


def get_conv_net(input_shape, num_classes, num_extra_conv_layers):
	model = Sequential()

	model.add(Conv2D(128, (3, 3), padding='same',
	                 input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(Activation('relu'))
	
	# model.add(Conv2D(64, (3, 3), padding='same'))
	# model.add(Activation('relu'))
	# model.add(Conv2D(64, (3, 3)))
	# model.add(Activation('relu'))
	
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	for i in range(num_extra_conv_layers):
		model.add(Conv2D(256, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(256, (3, 3), padding='same'))
		model.add(Activation('relu'))

		# model.add(Conv2D(128, (3, 3), padding='same'))
		# model.add(Activation('relu'))
		# model.add(Conv2D(128, (3, 3)))
		# model.add(Activation('relu'))

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])

	return model

# saved_model_v2
def get_conv_net_v2(input_shape, num_classes, num_extra_conv_layers):
	model = Sequential()

	model.add(Conv2D(64, (3, 3), padding='same',
	                 input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	
	# model.add(Conv2D(32, (3, 3), padding='same'))
	# model.add(Activation('relu'))
	# model.add(Conv2D(32, (3, 3)))
	# model.add(Activation('relu'))
	
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	for i in range(num_extra_conv_layers):
		model.add(Conv2D(128, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(128, (3, 3), padding='same'))
		model.add(Activation('relu'))

		# model.add(Conv2D(64, (3, 3), padding='same'))
		# model.add(Activation('relu'))
		# model.add(Conv2D(64, (3, 3)))
		# model.add(Activation('relu'))

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])

	return model

# saved_model_v1
def get_conv_net_small(input_shape, num_classes, num_extra_conv_layers):
	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same',
	                 input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	for i in range(num_extra_conv_layers):
		model.add(Conv2D(64, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])

	return model

