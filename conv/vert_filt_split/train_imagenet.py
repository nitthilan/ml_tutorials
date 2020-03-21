from keras_preprocessing.image import ImageDataGenerator
import keras
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import data_load.get_keras_data as gkd
import conv.networks.get_all_imagenet as gai

import numpy as np
from keras import layers
from keras import models
from keras.utils import to_categorical


network_type = sys.argv[1] #"VGG19" #"MobileNet"#"MobileNetV2"
cuda_device = sys.argv[2]
fraction = float(sys.argv[3])#1.0# 0.675


os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


num_classes = 1000
num_train_images = 1281166
num_val_images = 50000
num_test_images =  100000

img_dim = 224
# network_type = "MobileNet"#"VGG19" #"MobileNetV2"
num_epochs = 10
batch_size = 125

# def get_classifier_model():
	
# 	img_input = layers.Input(shape=input_shape)
# 	# Classification block
# 	x = layers.Flatten(name='flatten')(img_input)
# 	x = layers.Dense(4096, activation='relu', name='fc1')(x)
# 	x = layers.Dense(4096, activation='relu', name='fc2')(x)
# 	x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
# 	model = models.Model(img_input, x, name='vgg19')
# 	return model

def get_classifier_model(network_type, fraction):

	if(network_type == "VGG19"):
		input_shape = (7, 7, int(512*fraction))
		img_input = layers.Input(shape=input_shape)
		# Classification block
		x = layers.Flatten(name='flatten')(img_input)
		x = layers.Dense(int(4096*fraction), activation='relu', name='fc1')(x)
		x = layers.Dense(int(4096*fraction), activation='relu', name='fc2')(x)
		x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
		model = models.Model(img_input, x, name='vgg19')

	else:
		input_shape = (7, 7, int(1024*fraction))

		img_input = layers.Input(shape=input_shape)
		x = layers.GlobalAveragePooling2D()(img_input)
		shape = (1, 1, int(1024*fraction))
		dropout=1e-3
		x = layers.Reshape(shape, name='reshape_1')(x)
		x = layers.Dropout(dropout, name='dropout')(x)
		x = layers.Conv2D(num_classes, (1, 1),
		                  padding='same',
		                  name='conv_preds')(x)
		x = layers.Activation('softmax', name='act_softmax')(x)
		x = layers.Reshape((num_classes,), name='reshape_2')(x)

		model = models.Model(img_input, x, name='mobilenet') 
	return model


# parallel_model = multi_gpu_model(model, gpus=4)
parallel_model = get_classifier_model(network_type, fraction)

train_model = gai.get_all_nets(network_type, include_top=True)
train_model_1 = gai.get_all_nets(network_type, include_top=False)


train_model.summary()
parallel_model.summary()

train_model_weight_list = train_model.get_weights()
parallel_model_weight_list = parallel_model.get_weights()
print("Length of weights ", len(train_model_weight_list), len(parallel_model_weight_list))

if(network_type == "VGG19"):
	range_array = range(32, 38)
	offset = 32
else:
	range_array = range(135, 137)
	offset = 135

for i in range_array:
	print(parallel_model_weight_list[i-offset].shape, train_model_weight_list[i].shape)
	shape_vec = parallel_model_weight_list[i-offset].shape
	if(len(shape_vec)==2):
		parallel_model_weight_list[i-offset] = train_model_weight_list[i][:shape_vec[0], :shape_vec[1]]
	if(len(shape_vec)==1):
		parallel_model_weight_list[i-offset] = train_model_weight_list[i][:shape_vec[0]]

for i in range_array:
	print(i, offset)
	print(parallel_model_weight_list[i-offset].shape, train_model_weight_list[i].shape)
	shape_vec = parallel_model_weight_list[i-offset].shape
	if(len(shape_vec)==4):
		parallel_model_weight_list[i-offset] = train_model_weight_list[i][:,:,:shape_vec[2],:]
	else:
		parallel_model_weight_list[i-offset] = train_model_weight_list[i]

	# parallel_model_weight_list[i-offset] = train_model_weight_list[i]

parallel_model.set_weights(parallel_model_weight_list)
parallel_model_weight_list = parallel_model.get_weights()


train_model_1_weight_list = train_model_1.get_weights()
for j in range(offset):
	if((train_model_1_weight_list[j] == train_model_weight_list[j]).all()):
		print("Its a match ",j)
# for j in range_array:
# 	shape_vec = parallel_model_weight_list[i-offset].shape
# 	if((parallel_model_weight_list[j-offset] == train_model_weight_list[j]).all()):
# 		print("Its a match ",j)




save_weight_path = os.path.join("/mnt/additional/nitthilan/data/ml_tutorial/imagenet/", \
	network_type+str(fraction)+".h5")



modelCheckpoint = ModelCheckpoint(save_weight_path, 
  monitor='val_acc', verbose=0, save_best_only=True, 
  save_weights_only=False, mode='auto', period=1)

callbacks = [
            modelCheckpoint
            #   earlyStopping, 
            #   reduceonplateau,
            #   csv_logger
            ]

# initiate RMSprop optimizer
optimizers = keras.optimizers.rmsprop(lr=0.000001, decay=1e-6)
# optimizers = keras.optimizers.Adam(lr=0.001, decay=1e-6)


parallel_model.compile(loss='categorical_crossentropy', \
  optimizer=optimizers, metrics=['accuracy', 'top_k_categorical_accuracy'])

parallel_model.summary()

train_tf_path = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/val_"+network_type.lower()+".npz"
tf_output = np.load(train_tf_path)
print(tf_output["arr_0"].shape, tf_output["arr_1"].shape)

if(network_type == "VGG19"):
	num_filter_input = int(fraction*512)
else:
	num_filter_input = int(fraction*1024)

x_val = tf_output["arr_0"][:,:,:,:num_filter_input]
y_val_1 = tf_output["arr_1"]
y_val = to_categorical(y_val_1, num_classes=num_classes)

# parallel_model.load_weights(save_weight_path)
predict1 = parallel_model.evaluate(x_val, y_val)
print("Evaluate ", predict1)

parallel_model.save(save_weight_path)
# exit()

predict1 = parallel_model.predict(x_val, verbose=1) 
predict1 = np.argmax(predict1, axis=1)
# y_train = np.argmax(y_train, axis=1)

num_correct = 0
for pred, pred1 in zip(predict1,y_val_1):
	if(pred == pred1):
		num_correct += 1
	# print("Predcition ", pred, pred1)
print("Pre-predict ", num_correct/y_val_1.shape[0])



for j in range(200):
	for i in range(25, -1, -1):
		train_tf_path = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/train_"+network_type.lower()+"_"+str(i)+".npz"
		tf_output = np.load(train_tf_path)
		print(tf_output["arr_0"].shape, tf_output["arr_1"].shape)

		x_train = tf_output["arr_0"][:,:,:,:num_filter_input]
		y_train = tf_output["arr_1"]
		x_train = x_train[:y_train.shape[0]]
		y_train = to_categorical(y_train, num_classes=num_classes)
		print("Current iteration ",i,j, batch_size)

		parallel_model.fit(x_train, y_train,
	            batch_size=batch_size,
	            epochs=1,
	            validation_data=(x_val, y_val),
	            shuffle=True,
	            callbacks = callbacks)

		predict1 = parallel_model.predict(x_train)
		predict1 = np.argmax(predict1, axis=1)
		y_train = np.argmax(y_train, axis=1)

		num_correct = 0
		for pred, pred1 in zip(predict1,y_train):
			if(pred == pred1):
				num_correct += 1
			# print("Predcition ", pred, pred1)
		print(num_correct/y_train.shape[0])

		predict1 = parallel_model.predict(x_val)
		predict1 = np.argmax(predict1, axis=1)
		# y_train = np.argmax(y_train, axis=1)

		num_correct = 0
		for pred, pred1 in zip(predict1,y_val_1):
			if(pred == pred1):
				num_correct += 1
			# print("Predcition ", pred, pred1)
		print("Val accuracy ", num_correct/y_val_1.shape[0])



# train_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/train/"
# test_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/test/"
# val_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/val/"

# # train_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet-sz/224/train/"
# # test_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet-sz/test/"
# # val_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet-sz/224/val/"


# val_image_list = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/caffe_ilsvrc12.tar/val.txt"
# train_image_list = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/caffe_ilsvrc12.tar/train.txt"
# test_image_list = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/caffe_ilsvrc12.tar/test.txt"



# df_train=pd.read_csv(train_image_list, delim_whitespace=True, header=None, names=["filename", "class"])
# df_val=pd.read_csv(val_image_list, delim_whitespace=True, header=None, names=["filename", "class"])
# df_test=pd.read_csv(test_image_list, delim_whitespace=True, header=None, names=["filename", "class"])
# # df_train = df_train.apply(lambda x: x.astype(str).str.lower())
# # df_val = df_val.apply(lambda x: x.astype(str).str.lower())
# # df_test = df_test.apply(lambda x: x.astype(str).str.lower())


# print(df_test)
# print(df_val)
# print(df_train)

# train_generator = ImageDataGenerator().flow_from_dataframe(df_train, directory=train_base_folder, 
# 	x_col='filename', y_col='class', target_size=(img_dim, img_dim), has_ext=True, batch_size=batch_size)
# 	# , color_mode='rgb', classes=None, class_mode='categorical',  
# 	# shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png',
# 	# subset=None, interpolation='nearest', drop_duplicates=True)
# valid_generator = ImageDataGenerator().flow_from_dataframe(df_val, directory=val_base_folder, 
# 	x_col='filename', y_col='class', target_size=(img_dim, img_dim), has_ext=True, batch_size=batch_size)

# train_model = gai.get_all_nets(network_type, include_top=True)


# # weight_path = os.path.join(save_dir, \
# # 	model_name+str(num_filter)+".h5")
# # 	# model.save(weight_path)

# # modelCheckpoint = ModelCheckpoint(weight_path, 
# # 	monitor='val_acc', verbose=0, save_best_only=True, 
# # 	save_weights_only=False, mode='auto', period=1)

# callbacks = [
#           # modelCheckpoint
#           #   earlyStopping, 
#           #   reduceonplateau,
#           #   csv_logger
#           ]

# # initiate RMSprop optimizer
# # optimizers = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# # optimizers = keras.optimizers.Adam(lr=0.01, decay=1e-6)
# optimizers = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# train_model.compile(loss='categorical_crossentropy', \
# 	optimizer=optimizers, metrics=['accuracy'])


# for i in range(int((num_train_images+49999)/50000)):
# 	train_generator = ImageDataGenerator().flow_from_dataframe(df_train.iloc[i*50000:(i+1)*50000,:].reset_index(drop=True), 
# 		directory=train_base_folder, 
# 		x_col='filename', y_col='class', target_size=(img_dim, img_dim), has_ext=True, batch_size=batch_size)


# 	prediction = train_model.predict_generator(train_generator,
# 	                  steps=50000 // batch_size,
# 	                  verbose=1)

# 	print(prediction.shape[0])
# 	np.savez(train_tf_path, prediction, df_train['class'][i*50000:(i+1)*50000])

# # npzfile = np.load(train_tf_path)

# # print(npzfile['arr_1'].shape, npzfile['arr_0'].shape)


# prediction = train_model.predict_generator(valid_generator,
#                   steps=num_val_images // batch_size,
#                   verbose=1)

# print(prediction.shape)

# train_tf_path = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/val_vgg19.npz"
# np.savez(train_tf_path, prediction, df_val['class'][:prediction.shape[0]])


