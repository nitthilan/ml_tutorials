
import os
from os import listdir
from os.path import isfile, join
import pydicom, numpy as np
import time
from PIL import Image

base_folder = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/"

imagenet_folder = "/mnt/additional/aryan/imagenet_validation_data/ILSVRC2012_img_val/"

df_preprocess_base_folder = base_folder
df_val_preprocess_filename = df_preprocess_base_folder + "val_10000.npz"
# df_test_preprocess_filename = df_preprocess_base_folder + "test.npz"


def read_images(basefolder, output_filename, size, filerange):
	(w, h) = size
	(start, end) = filerange
	print("Preprocessing details ", start, end, output_filename)
	filelist = [f for f in listdir(basefolder) \
		if isfile(join(basefolder, f))]
	filename_list = []
	print("Total files ", len(filelist))
	image_array_np = np.zeros((end-start, w, h, 3))
	for i, file in enumerate(filelist[start:end]):
		# ds = pydicom.dcmread(basefolder+file) 
		image = Image.open(basefolder+file)
		image = image.convert('RGB')
		width, height = image.size
		pixel_values = list(image.getdata())
		if image.mode == 'RGB':
			channels = 3
		elif image.mode == 'L':
			channels = 1
		else:
			print("Unknown mode: %s" % image.mode, file)

			return None
		pixel_values = image.resize((w,h), Image.ANTIALIAS)
		pixel_values = np.array(pixel_values).reshape((w, h, channels))
		# print(pixel_values.shape, pixel_values.dtype,
		# 	np.min(pixel_values), np.max(pixel_values))
		image_array_np[i] = pixel_values
		if(i%1000 == 0):
			print("Num Processed ",i)
		# print(dir(pixel_values))
		filename_list.append(file)

	print(image_array_np.shape)
	np.savez(output_filename, image_array=image_array_np, 
		filename_list=filename_list)
	return image_array_np, filename_list

# for i in [3]:#range(5):
# 	filerange = (i*10000, (i+1)*10000)
# 	df_val_preprocess_filename = df_preprocess_base_folder+"val_"+str(i*10000)+".npz"
# 	read_images(imagenet_folder, df_val_preprocess_filename, (224, 224), filerange)

for i in [4]:#range(5):
	df_val_preprocess_filename = df_preprocess_base_folder+"val_"+str(i*10000)+".npz"
	df_val_preprocess_filename1 = df_preprocess_base_folder+"val1_"+str(i*10000)+".npz"
	filevalue = np.load(df_val_preprocess_filename)
	np.savez(df_val_preprocess_filename1, image_array=filevalue["image_array"][:10000], 
		filename_list=filevalue["filename_list"])


# image = Image.open(imagenet_folder+"ILSVRC2012_val_00019877.JPEG")
# image = image.convert('RGB')
# width, height = image.size
# # pixel_values = list(image.getdata())
# # pixel_values = image.resize((w,h), Image.ANTIALIAS)
# # print(pixel_values.shape)
# # pixel_values = np.array(pixel_values).reshape((w, h, channels))
# if image.mode == 'RGB':
# 	channels = 3
# elif image.mode == 'L':
# 	channels = 1
# else:
# 	print("Unknown mode: %s" % image.mode)