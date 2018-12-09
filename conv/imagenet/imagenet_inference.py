
import os
import numpy as np
import csv

import get_all_imagenet as gai
import time
from keras.utils import to_categorical
from keras import optimizers


base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/"

val_true_pred_path = "caffe_ilsvrc12.tar/val.txt"
val_images_file = "val1_0.npz"

val_true_pred_file_info_path = "val_true_pred_file_info.npz"

def get_val_images(filepath):
	filevalue = np.load(filepath)
	return (filevalue["image_array"], filevalue["filename_list"])

def get_val_true_pred_info(filepath):
	filevalue = np.load(filepath)
	return (filevalue["val_true_pred_list"], filevalue["filename_list"])

def get_val_true_pred(val_filelist, val_true_pred_path):
	val_filelist_idx = {}
	with open(val_true_pred_path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=' ')
		for row in csv_reader:
			# print("Row ", row)
			val_filelist_idx[row[0]] = int(row[1])
	
	true_pred_list = []
	for file in val_filelist:
		# print("File list ", file)
		true_pred_list.append(val_filelist_idx[file])

	return np.array(true_pred_list)

def split_train_val(input_images, split_percentage, num_classes=1000):
	num_train_image = int(split_percentage*50)
	num_val_image = 50 - num_train_image
	train_data = np.zeros((num_classes*num_train_image, 224, 224, 3), dtype=np.uint8)
	val_data = np.zeros((num_classes*num_val_image, 224, 224, 3), dtype=np.uint8)
	train_true = np.zeros(num_classes*num_train_image)
	val_true = np.zeros(num_classes*num_val_image)
	# Fill train
	for i in range(num_train_image):
		for j in range(num_classes):
			train_data[i*num_classes+j] = input_images[j, i,:,:]
			train_true[i*num_classes+j] = j

	# Fill validation
	for i in range(num_train_image, 50):
		for j in range(num_classes):
			val_data[(i-num_train_image)*num_classes+j] = input_images[j, i,:,:]
			val_true[(i-num_train_image)*num_classes+j] = j
	return (train_data, val_data, train_true, val_true)


def get_mask_weights(weight_list_copy, mask):
	for i,weight in enumerate(weight_list_copy):
		if(len(weight.shape) == 4 and i != 0):
			if(weight.shape[0] == 3 and weight.shape[1] == 3):
				for j in range(int(mask*weight.shape[2]/4), weight.shape[2]):
					weight[:,:,j,:] = 0
			if(weight.shape[0] == 1 and weight.shape[1] == 1):
				for j in range(int(mask*weight.shape[3]/4), weight.shape[3]):
					weight[:,:,:,j] = 0
			print("Layer modified ", i, int(mask*weight.shape[2]/4), weight.shape[2],
				(mask*weight.shape[3]/4), weight.shape[3])

			# if(weight.shape[0] == 3 and weight.shape[1] == 3):
			# 	for j in range(weight.shape[2]-mask, weight.shape[2]):
			# 		weight[:,:,j,:] = 0
			# if(weight.shape[0] == 1 and weight.shape[1] == 1):
			# 	for j in range(weight.shape[3]-mask, weight.shape[3]):
			# 		weight[:,:,:,j] = 0
		print("Weight shape ", i, weight.shape)
	return weight_list_copy

def run_inference(val_file="full_val.npz"):
# for i, val_file in enumerate(["val1_0.npz"]):#, "val1_10000.npz", "val1_20000.npz", "val1_30000.npz", "val1_40000.npz"]):
	start_time = time.time()
	(val_images_np, val_filelist) = \
		get_val_images(os.path.join(base_folder, val_file))

	print("Total Time ", time.time() - start_time)

	split_percentage = 0.5
	num_output = 100
	val_true_pred_list = get_val_true_pred(val_filelist, 
		os.path.join(base_folder, val_true_pred_path))
	(train_data, val_data, train_true, val_true) = \
		split_train_val(val_images_np, split_percentage, num_output)

	# np.savez(os.path.join(base_folder, val_true_pred_file_info_path), 
	# 	val_true_pred_list=val_true_pred_list, 
	# 	filename_list=val_filelist)

	# val_true_pred_list, val_filelist = \
	# 	get_val_true_pred_info(os.path.join(base_folder, val_true_pred_file_info_path))
	# print("Pred Data size ", val_true_pred_list.shape)

	train_true_cat = to_categorical(train_true)
	val_true_cat = to_categorical(val_true)
	print("To category ", np.argmax(train_true_cat, axis=1))

	print("True Train ", train_true)

	print("Image details ", train_data.shape, train_true.shape, 
		val_data.shape, val_true.shape, 
		train_true_cat.shape, val_true_cat.shape)

	# model_name = 
	for model_name in ["MobileNet"]: #, "MobileNetV2", "VGG19", "ResNet50"]:
		model = gai.get_all_nets(model_name, include_top=False)
		train_data_preproc = gai.preprocess_image(model_name, np.copy(train_data))
		val_data_preproc = gai.preprocess_image(model_name, np.copy(val_data))

		# pred_output = model.predict(val_images_np)
		# score = model.evaluate(train_data_preproc , train_true_cat, 
		# 	batch_size=32, verbose=1)
		# print("Score ", model_name, val_file, score)
		# score = model.evaluate(val_data_preproc , val_true_cat, 
		# 	batch_size=32, verbose=1)
		# print("Score ", model_name, val_file, score)
		# pred_value = model.predict(train_data_preproc, batch_size=1, verbose=1)
		# print("Pred value ", np.argmax(pred_value, axis=1))
		# print(train_true)
		weight_list = model.get_weights()

		for mask in [1, 2, 3]:
			weight_list_copy = np.copy(weight_list)#copy.deepcopy(weight_list)
			weight_list_copy = get_mask_weights(weight_list_copy, mask)
			model.set_weights(weight_list_copy)
			model =	gai.add_classifier(model, num_output)
			
			# pred_output = model.predict(val_images_np)
			# score = model.evaluate(val_images_np_1 , val_true_pred_list_1, 
			# 	batch_size=1, verbose=1)
			# print("Score ", model_name, val_file, mask, score)
			batch_size = 32
			epochs = 100

			# for layer in model.layers[:-3]:
			# 	layer.trainable = False
			# for layer in model.layers[-3:]:
			# 	layer.trainable = True
			opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

			# Let's train the model using RMSprop
			model.compile(loss='categorical_crossentropy',
			      optimizer=opt,
			      metrics=['accuracy'])
			model.summary()

			# pred_value = model.predict(val_images_np, batch_size=1, verbose=1)
			# print("Pred value ", np.argmax(pred_value, axis=1))
			history = model.fit(train_data_preproc, train_true_cat,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(val_data_preproc, val_true_cat),
                shuffle=True)

run_inference()
