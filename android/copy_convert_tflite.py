from shutil import copyfile
import tensorflow as tf

# # Copying files
# models_list = ["conv"]#, "vgg", "MobileNet"]
# datasets_list = ["svhn"]#, "mnist", "cifar10"]
# for model_name in models_list:
# 	for dataset in datasets_list:
# 		for num_filter in [2.0, 2.25, 2.5, 3.0, 3.5, 4.0]:
			
# 			if(num_filter == 4):
# 				src_base_path = "./conv/vert_filt_saved_keras_models/"
# 				src_file_name = model_name+"_"+dataset+"_un_4.0.h5"
# 			else:
# 				src_base_path = "./conv/saved_models/"
# 				src_file_name = "vert_filt_wbn_"+model_name+"_"+dataset+"_"+str(num_filter)+".h5"

# 			dest_base_path = "./conv/vert_filt_models/"
# 			dest_file_name = "vert_filt_"+model_name+"_"+dataset+"_"+str(num_filter)+".h5"
# 			if(num_filter == 2.25 and dataset != "cifar10"):
# 				continue
# 			# if(model_name == "conv" and dataset == "svhn"):
# 			# 	continue

# 			print(src_base_path+src_file_name, dest_base_path+dest_file_name)
# 			copyfile(src_base_path+src_file_name, dest_base_path+dest_file_name)

# # Use mobilenet_for_mobile_conversion.py to conver mobilenet to mobilenet for mobile 
# models_list = ["conv"]#, "vgg", "MobileNet_for_mobile"]
# datasets_list = ["svhn"]#, "mnist", "cifar10"]
# for model_name in models_list:
# 	for dataset in datasets_list:
# 		for num_filter in [2.0, 2.25, 2.5, 3.0, 3.5, 4.0]:
			
# 			dest_base_path = "./conv/vert_filt_models/"
# 			dest_file_name = "vert_filt_"+model_name+"_"+dataset+"_"+str(num_filter)+".h5"
# 			if(num_filter == 2.25 and dataset != "cifar10"):
# 				continue
# 			# if(model_name == "conv" and dataset == "svhn"):
# 			# 	continue

# 			input_keras_model = dest_base_path+dest_file_name
# 			output_tflite_model = input_keras_model[:-3]+".tflite"
# 			converter = tf.contrib.lite.TocoConverter.from_keras_model_file(input_keras_model)
# 			tflite_model = converter.convert()
# 			open(output_tflite_model, "wb").write(tflite_model)


models_list = ["conv", "vgg", "MobileNet_for_mobile"]
datasets_list = ["svhn", "mnist", "cifar10"]
for model_name in models_list:
	for dataset in datasets_list:
		for num_filter in [2.0, 2.25, 2.5, 3.0, 3.5, 4.0]:
			
			dest_base_path = "./conv/vert_filt_models/"
			dest_file_name = "vert_filt_"+model_name+"_"+dataset+"_"+str(num_filter)+".h5"
			if(num_filter == 2.25 and dataset != "cifar10"):
				continue
			# if(model_name == "conv" and dataset == "svhn"):
			# 	continue

			input_keras_model = dest_file_name
			output_tflite_model = input_keras_model[:-3]+".tflite"
			# generate_command(output_tflite_model)
			print("adb push /mnt/additional/nitthilan/ml_tutorials/conv/vert_filt_models/"+output_tflite_model+" /data/local/tmp")


for model_name in models_list:
	for dataset in datasets_list:
		for num_filter in [2.0, 2.25, 2.5, 3.0, 3.5, 4.0]:
			
			dest_base_path = "./conv/vert_filt_models/"
			dest_file_name = "vert_filt_"+model_name+"_"+dataset+"_"+str(num_filter)+".h5"
			if(num_filter == 2.25 and dataset != "cifar10"):
				continue
			# if(model_name == "conv" and dataset == "svhn"):
			# 	continue

			input_keras_model = dest_file_name
			output_tflite_model = input_keras_model[:-3]+".tflite"
			# generate_command(output_tflite_model)
			print("adb shell taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/"+output_tflite_model+" --num_threads=1")

			
