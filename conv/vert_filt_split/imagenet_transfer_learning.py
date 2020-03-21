from keras_preprocessing.image import ImageDataGenerator
import keras
import pandas as pd
import tensorflow as tf
from keras import backend as K
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import data_load.get_keras_data as gkd
import conv.networks.get_all_imagenet as gai

# from keras.applications.mobilenet import preprocess_input

from keras.applications import mobilenet
from keras.applications import vgg19


import numpy as np
from keras import layers
from keras import models
from keras.models import Model
from keras.utils import to_categorical




network_type = sys.argv[1] #"VGG19" #"MobileNet"#"MobileNetV2"
cuda_device = sys.argv[2]

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device#"1"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

train_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/train/"
test_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/test/"
val_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/val/"

# train_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet-sz/224/train/"
# test_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet-sz/test/"
# val_base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet-sz/224/val/"


val_image_list = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/caffe_ilsvrc12.tar/val.txt"
train_image_list = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/caffe_ilsvrc12.tar/train.txt"
test_image_list = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/caffe_ilsvrc12.tar/test.txt"
num_train_images = 1281166
num_val_images = 50000
num_test_images =  100000
num_classes = 1000

img_dim = 224
num_epochs = 10
batch_size = 125



df_train=pd.read_csv(train_image_list, delim_whitespace=True, header=None, names=["filename", "class"])
df_val=pd.read_csv(val_image_list, delim_whitespace=True, header=None, names=["filename", "class"])
df_test=pd.read_csv(test_image_list, delim_whitespace=True, header=None, names=["filename", "class"])
# df_train = df_train.apply(lambda x: x.astype(str).str.lower())
# df_val = df_val.apply(lambda x: x.astype(str).str.lower())
# df_test = df_test.apply(lambda x: x.astype(str).str.lower())


# print(df_test)
# print(df_val)
# print(df_train)

# preprocess_input = vgg19.preprocess_input
# # preprocess_input = mobilenet.preprocess_input
preprocess_input = gai.preprocess_image_fn(network_type)

def get_classifier_model(network_type):

	if(network_type == "VGG19"):
		# input_shape = (25088,)
		input_shape = (7, 7, 512)
		img_input = layers.Input(shape=input_shape)
		# Classification block
		x = layers.Flatten(name='flatten')(img_input)
		x = layers.Dense(4096, activation='relu', name='fc1')(x)
		x = layers.Dense(4096, activation='relu', name='fc2')(x)
		x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
		model = models.Model(img_input, x, name='vgg19')

	else:
		input_shape = (7,7, 1024)

		img_input = layers.Input(shape=input_shape)
		x = layers.GlobalAveragePooling2D()(img_input)
		shape = (1, 1, int(1024))
		# shape = (int(1024), 1, 1)
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

# train_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(df_train,
# 	directory=train_base_folder, 
# 	x_col='filename', y_col='class', shuffle=False, target_size=(img_dim, img_dim), has_ext=True, batch_size=batch_size)
# valid_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(df_val, 
# 	directory=val_base_folder, 
# 	x_col='filename', y_col='class', shuffle=False, target_size=(img_dim, img_dim), has_ext=True, batch_size=batch_size)

# train_generator = ImageDataGenerator().flow_from_dataframe(df_train, directory=train_base_folder, 
# 	x_col='filename', y_col='class', target_size=(img_dim, img_dim), has_ext=True, batch_size=batch_size)
# 	# , color_mode='rgb', classes=None, class_mode='categorical',  
# 	# shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png',
# 	# subset=None, interpolation='nearest', drop_duplicates=True)



# weight_path = os.path.join(save_dir, \
# 	model_name+str(num_filter)+".h5")
# 	# model.save(weight_path)

# modelCheckpoint = ModelCheckpoint(weight_path, 
# 	monitor='val_acc', verbose=0, save_best_only=True, 
# 	save_weights_only=False, mode='auto', period=1)

callbacks = [
          # modelCheckpoint
          #   earlyStopping, 
          #   reduceonplateau,
          #   csv_logger
          ]

# initiate RMSprop optimizer
# optimizers = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# optimizers = keras.optimizers.Adam(lr=0.01, decay=1e-6)
optimizers = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

train_model = gai.get_all_nets(network_type, include_top=True) 

train_model.summary()
exit()

train_model_1 = gai.get_all_nets(network_type, include_top=False)# Model(input=train_model.input, output=[train_model.layers[-5].output])#

input_shape = (25088,)

img_input = layers.Input(shape=input_shape)
classifier = get_classifier_model(network_type) # Model(input=train_model.layers[-3].input, output=[train_model.layers[-1].output])#

train_model.summary()
classifier.summary()

# for i in range(3):
# 	train_model_1.layers.pop()
train_model_1.summary()

train_model_weight_list = train_model.get_weights()
parallel_model_weight_list = classifier.get_weights()

if(network_type == "VGG19"):
	range_array = range(32, 38)
	offset = 32
else:
	range_array = range(135, 137)
	offset = 135

print("Length of weights ", len(train_model_weight_list), len(parallel_model_weight_list),
	len(train_model_1.get_weights()), range_array)

for i in range_array:
	print(i, offset)
	parallel_model_weight_list[i-offset] = train_model_weight_list[i]

classifier.set_weights(parallel_model_weight_list)
parallel_model_weight_list = classifier.get_weights()


num_steps = 50000 // batch_size#10#

train_model.compile(loss='categorical_crossentropy', \
	optimizer=optimizers, metrics=['accuracy'])


for i in range(int((num_train_images+49999)/50000)-1, -1, -1):
	train_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(
		df_train.iloc[i*50000:(i+1)*50000,:].reset_index(drop=True), 
		directory=train_base_folder, shuffle=False,
		x_col='filename', y_col='class', target_size=(img_dim, img_dim), has_ext=True, batch_size=batch_size)


	prediction = train_model.predict_generator(train_generator,
	                  steps=num_steps,
	                  verbose=1)

	print(prediction.shape[0])

	prediction = np.argmax(prediction, axis=1)
	y_train = df_train['class'][i*50000:(i+1)*50000]#np.argmax(y_train, axis=1)

	num_correct = 0
	for pred, pred1 in zip(prediction,y_train):
		if(pred == pred1):
			num_correct += 1
		# print("Predcition ", pred, pred1)
	print(num_correct*1.0/prediction.shape[0])

	train_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(
		df_train.iloc[i*50000:(i+1)*50000,:].reset_index(drop=True), 
		directory=train_base_folder, shuffle=False,
		x_col='filename', y_col='class', target_size=(img_dim, img_dim), has_ext=True, batch_size=batch_size)

	prediction_1 = train_model_1.predict_generator(train_generator,
	                  steps=num_steps,
	                  verbose=1)
	train_tf_path = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/train_"+network_type.lower()+"_"+str(i)+".npz"
	np.savez(train_tf_path, prediction_1, df_train['class'][i*50000:(i+1)*50000])


	predict1 = classifier.predict(prediction_1, verbose=1)
	predict1 = np.argmax(predict1, axis=1)

	# num_correct = 0
	# for pred, pred1 in zip(prediction,predict1):
	# 	if(pred == pred1):
	# 		num_correct += 1
	# 	# print("Predcition1 ", pred, pred1)
	# print(num_correct*1.0/prediction.shape[0])

	y_train = df_train['class'][i*50000:(i+1)*50000]#np.argmax(y_train, axis=1)

	num_correct = 0
	for pred, pred1 in zip(predict1,y_train):
		if(pred == pred1):
			num_correct += 1
		# print("Predcition ", pred, pred1)
	print(num_correct*1.0/predict1.shape[0])

# npzfile = np.load(train_tf_path)

# print(npzfile['arr_1'].shape, npzfile['arr_0'].shape)


train_tf_path = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/val_"+network_type.lower()+".npz"

valid_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_dataframe(
	df_val, directory=val_base_folder, shuffle=False,
	x_col='filename', y_col='class', target_size=(img_dim, img_dim), has_ext=True, batch_size=batch_size)

prediction = train_model_1.predict_generator(valid_generator,
                  steps=num_val_images // batch_size,
                  verbose=1)

print(prediction.shape)
np.savez(train_tf_path, prediction, df_val['class'][:prediction.shape[0]])


# tf_output = np.load(train_tf_path)
# print(tf_output["arr_0"].shape, tf_output["arr_1"].shape)
# prediction = tf_output["arr_0"]
# y_val = tf_output["arr_1"]
# classifier.compile(loss='categorical_crossentropy', \
# 	optimizer=optimizers, metrics=['accuracy'])
# y_val_1 = to_categorical(y_val, num_classes=num_classes)

# predict1 = classifier.evaluate(prediction, y_val_1)
# print("Evaluate ", predict1)


predict1 = classifier.predict(prediction, verbose=1) 

predict1 = np.argmax(predict1, axis=1)

# num_correct = 0
# for pred, pred1 in zip(prediction,predict1):
# 	if(pred == pred1):
# 		num_correct += 1
# 	# print("Predcition1 ", pred, pred1)
# print(num_correct*1.0/prediction.shape[0])

# y_val = df_val['class'][:prediction.shape[0]]#np.argmax(y_train, axis=1)

num_correct = 0
for pred, pred1 in zip(predict1, y_val):
	if(pred == pred1):
		num_correct += 1
	# print("Predcition ", pred, pred1)
print(num_correct*1.0/predict1.shape[0])


