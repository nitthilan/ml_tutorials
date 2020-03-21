

import os
import numpy as np
import csv
import copy

import get_all_imagenet as gai
import time
from keras.utils import to_categorical
import h5py


# https://github.com/wichtounet/frameworks/blob/master/keras/experiment6.py
base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/"

val_true_pred_path = "caffe_ilsvrc12.tar/val.txt"



def get_val_images(filepath):
	filevalue = np.load(filepath)
	return (filevalue["image_array"], filevalue["filename_list"])

def get_val_true_pred_info(filepath):
	filevalue = np.load(filepath)
	return (filevalue["val_true_pred_list"], filevalue["filename_list"])

def get_val_true_pred(val_true_pred_path):
	val_filelist_idx = {}
	histogram = np.zeros(1000)
	with open(val_true_pred_path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=' ')
		for i, row in enumerate(csv_reader):
			# print("Row ", row)
			val_filelist_idx[row[0]] = int(row[1])
			histogram[int(row[1])] += 1
			# if(i%100):
			# 	print(histogram)

	
	print(histogram)
	return val_filelist_idx

val_filelist_idx = get_val_true_pred(os.path.join(base_folder, val_true_pred_path))
# hf = h5py.File(os.path.join(base_folder, 'dataset.h5'), 'w')
# group_list = []
# for i in range(1000):
# 	group_list.append(hf.create_group(str(i)))
# hf.close()

val_data = np.zeros((1000, 50, 224, 224, 3), dtype=np.uint8)
histogram = np.zeros(1000, dtype=np.uint8)
val_filelist = []
for i, val_file in enumerate(["val1_0.npz", "val1_10000.npz", "val1_20000.npz", "val1_30000.npz", "val1_40000.npz"]):
	start_time = time.time()
	val_file_path = os.path.join(base_folder, val_file)
	image_array, filename_list = get_val_images(val_file_path)
	print("Total time ", time.time() - start_time, val_file)
	# np.savez(os.path.join(base_folder, val_file), 
	# 	image_array=image_array.astype(np.uint8), 
	# 	filename_list=filename_list)
	for j,filename in enumerate(filename_list):
		# print("Size of data ", image_array.shape)
		# group_list[val_filelist_idx[filename]].create_dataset(filename, image_array[j])
		index = int(val_filelist_idx[filename])
		val_data[index, histogram[index],:,:] = image_array[j]
		histogram[index] += 1
		val_filelist.append(filename)

np.savez(os.path.join(base_folder, "full_val.npz"), 
		image_array=val_data.astype(np.uint8), 
		filename_list=val_filelist)