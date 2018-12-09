
import os
import numpy as np
import csv
import copy

import get_all_imagenet as gai
import time
from keras.utils import to_categorical


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

for i, val_file in enumerate(["val1_0.npz"]):#, "val1_10000.npz", "val1_20000.npz", "val1_30000.npz", "val1_40000.npz"]):
	start_time = time.time()
	(val_images_np, val_filelist) = \
		get_val_images(os.path.join(base_folder, val_file))

	print("Total Time ", time.time() - start_time)

	val_true_pred_list = get_val_true_pred(val_filelist, 
		os.path.join(base_folder, val_true_pred_path))

	# np.savez(os.path.join(base_folder, val_true_pred_file_info_path), 
	# 	val_true_pred_list=val_true_pred_list, 
	# 	filename_list=val_filelist)

	# val_true_pred_list, val_filelist = \
	# 	get_val_true_pred_info(os.path.join(base_folder, val_true_pred_file_info_path))
	# print("Pred Data size ", val_true_pred_list.shape)

	val_true_pred_list_1 = to_categorical(val_true_pred_list)
	print("To category ", np.argmax(val_true_pred_list_1, axis=1))

	print("Prediction ", val_true_pred_list)

	print("Image details ", val_images_np.shape, len(val_filelist), val_true_pred_list_1.shape)

	num_train = int(10000*0.8)
	x_train = val_images_np[:num_train]
	y_train = val_true_pred_list_1[:num_train]
	x_test = val_images_np[num_train:]
	y_test = val_true_pred_list_1[num_train:]

	# model_name = 
	for model_name in ["MobileNet"]: #, "VGG19", "MobileNetV2", "ResNet50"]:
		model = gai.get_all_nets(model_name)
		val_images_np_1 = gai.preprocess_image(model_name, np.copy(val_images_np))

		weight_list = model.get_weights()

		for mask in [1, 2, 3]:
			weight_list_copy = copy.deepcopy(weight_list)
			weight_list_copy = get_mask_weights(weight_list_copy, mask)
			model.set_weights(weight_list_copy)
			model.summary()

			# pred_output = model.predict(val_images_np)
			# score = model.evaluate(val_images_np_1 , val_true_pred_list_1, 
			# 	batch_size=1, verbose=1)
			# print("Score ", model_name, val_file, mask, score)
			batch_size = 32
			epochs = 100

			for layer in model.layers[:-1]:
				layer.trainable = False

			# pred_value = model.predict(val_images_np, batch_size=1, verbose=1)
			# print("Pred value ", np.argmax(pred_value, axis=1))
			history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True)

