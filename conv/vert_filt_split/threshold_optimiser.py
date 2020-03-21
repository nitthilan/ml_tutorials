import numpy as np 
from bayes_opt import BayesianOptimization
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import networks.confidence as cf
import data_load.get_keras_data as gkd

from PIL import Image


def get_pred_y_test(models_path_list):
	predict_gen_list = []
	y_test_list = []
	# load all the prediction and true value
	for pred_path in models_path_list:
		predict_path = pred_path+"_predict.npz"
		# predict_path = pred_path[:-3]+"_predict.npz"
		loaded_data = np.load(predict_path)
		predict_gen = loaded_data['arr_0']
		y_test = loaded_data['arr_1']
		# print("Predicted value ", y_test.shape, predict_gen.shape)
		predict_gen_list.append(predict_gen)
		y_test_list.append(y_test)
	return predict_gen_list, y_test_list


def get_accuracy(num_models, ctype, predict_gen_list, y_test_list):
	accuracy_values = {}
	min_acc = 100
	max_acc = 0
	for idx in range(num_models):
		(accuracy, total_pred, conf_sel_list) = \
			cf.get_prob_based_confidence(\
				predict_gen_list[idx], \
				y_test_list[idx], \
	  			0.0, ctype)
		accuracy_values[idx] = accuracy*1.0/total_pred
		# print(accuracy, total_pred, accuracy_values[idx])

		# Min and max accuaracy
		if(min_acc > accuracy_values[idx]):
			min_acc = accuracy_values[idx]
		if(max_acc < accuracy_values[idx]):
			max_acc = accuracy_values[idx]
	return accuracy_values, min_acc, max_acc

def dump_images(test_images, output_idx):
	# print(test_images.shape[0])
	if(test_images.shape[0]<25):

		return

	N = 4
	W = 6
	image_tile = np.zeros((N*32,W*32,3))
	for i in range(N):
		for j in range(W):
			image_tile[i*32:(i+1)*32,j*32:(j+1)*32] = test_images[i*10+j]

	# image_tile = 255*image_tile
	# print(image_tile.astype('uint8'))

	im = Image.fromarray(image_tile.astype('uint8'))
	im.save("image_tile_"+str(output_idx)+".png")

	return

class Optimiser:
	def __init__(self, optimiser_value, models_path_list, is_return_norm):
		self.optimiser_value = optimiser_value
		self.models_path_list = models_path_list
		self.dict_to_conf_map = {
			0:"max_prob", 1:"top_pred_diff", 2:"entropy"
		}
		self.is_return_norm = is_return_norm;
		return
	def print_model_stats(self, ctype):
		ctype = self.dict_to_conf_map[int(ctype)]
		predict_gen_list, y_test_list = \
			get_pred_y_test(self.models_path_list)
		accuracy_values, min_acc, max_acc = \
			get_accuracy(len(self.models_path_list),\
				ctype, predict_gen_list, y_test_list)

		max_value = self.optimiser_value[0]
		accu_list = []
		for i in range(len(self.optimiser_value)):
			accu_list.append(accuracy_values[i])

		print(np.array(accu_list), np.array(self.optimiser_value)/max_value)

	def find_acc_energy(self, c1, c2, c3, c4, ctype):

		# x_train, y_train, x_test, y_test = \
	 #    	gkd.get_data("cifar10")
		# print(y_test.shape)

		max_value = self.optimiser_value[0]
		min_value = self.optimiser_value[-1] #(3.3+2.16+1)
		# print(max_value, min_value)
		ctype = self.dict_to_conf_map[int(ctype)]
		predict_gen_list, y_test_list = \
			get_pred_y_test(self.models_path_list)
		accuracy_values, min_acc, max_acc = \
			get_accuracy(len(self.models_path_list),\
				ctype, predict_gen_list, y_test_list)
		num_test_vectors = y_test_list[0].shape[0]
		# print("Accuracy ", min_acc, max_acc, accuracy_values)

		confidence_list = [0,c1,c2,c3,c4]
		accuracy_dict = {}
		total_pred_dict = {}
		test_image_dict = {}
		for idx in range(len(self.optimiser_value)-1, -1, -1):
			(accuracy, total_pred, conf_sel_list) = \
				cf.get_prob_based_confidence(\
					predict_gen_list[idx], \
					y_test_list[idx], \
	  				confidence_list[idx], \
	  				ctype)
			accuracy_dict[idx] = accuracy
			total_pred_dict[idx] = total_pred
			# print("Iteration ", idx)

			# 
			# dump_images(x_test[conf_sel_list], idx)
			conf_sel_list = np.logical_not(conf_sel_list)
			# x_test = x_test[conf_sel_list]

			# print(conf_sel_list)

			for idx1 in range(idx):
				predict_gen_list[idx1] = predict_gen_list[idx1][conf_sel_list]
				y_test_list[idx1] = y_test_list[idx1][conf_sel_list]

		# Total Accuracy
		total_act = 0
		value_act = 0
		for idx in range(len(self.optimiser_value)):
			total_act += accuracy_dict[idx]
			value_act += (total_pred_dict[idx]*self.optimiser_value[idx])
			# value_act += (0.25*total_pred_dict[idx]*(len(self.optimiser_value) - idx - 1)*self.optimiser_value[-1])
		# total_act /= 10000.0#26032.0
		# value_act /= 10000.0#26032.0
		total_act /= (1.0*num_test_vectors)
		value_act /= (1.0*num_test_vectors)
		total_acc_nor = (total_act - min_acc)*1.0/(max_acc - min_acc)
		
		value_spent = (value_act - min_value)*1.0/(max_value - min_value)
		value_spent_nor = 1 - value_spent
		# print(value_act, value_spent_nor, total_act, total_acc_nor)
		print(accuracy_dict, total_pred_dict)
		# print(ld*total_acc_nor + (1-ld)*(value_spent_nor))

		print_list = [c1, value_act/max_value, total_act]
		print_list = [str(a) for a in print_list]
		# print('['+', '.join(print_list)+'],')
		if(self.is_return_norm):
			return total_acc_nor, value_spent_nor
		else:
			return total_act, value_act/max_value

	def find_optimal_param(self, c1, c2, c3, c4, ctype, ld):

		total_acc_nor, value_spent_nor = \
			self.find_acc_energy(c1, c2, c3, c4, ctype)

		return ld*total_acc_nor + (1-ld)*(value_spent_nor)



# Order the models based on the decreasing accuracy or size
load_conv_cifar10_list = [
	"./vert_filt_saved_keras_models/conv_cifar10_false_4.h5",
	"./saved_models_bkup/vert_filt_conv_cifar10_3.h5",
	"./saved_models_bkup/vert_filt_conv_cifar10_2.h5",	
]
time_conv_cifar10_list = [3.373, 2.02, 1.0]#[36854, 22168, 10926]#[74, 56, 35]
time2_conv_cifar10_list = [11.38, 4.08, 1.0]#[36854, 22168, 10926]#[74, 56, 35]

load_cifar10_vgg_list = [
	"./vert_filt_saved_keras_models/vgg_cifar10_false_4.h5",
	"./saved_models_bkup/vert_filt_vgg16_cifar10_3.h5",
	"./saved_models_bkup/vert_filt_vgg16_cifar10_2.h5",	
	# "./saved_models_bkup/vert_filt_vgg16_cifar10_4.h5",
]
time_cifar10_vgg_list = [2.42, 1.37, 1.0]#[85210, 48287, 35087]#[105, 89, 77]
time2_cifar10_vgg_list = [5.86, 1.88, 1.0]#[85210, 48287, 35087]#[105, 89, 77]

load_conv_mnist_list = [
	"./vert_filt_saved_keras_models/conv_mnist_false_4.h5",
	"./saved_models_bkup/vert_filt_conv_mnist_3.h5",	
	"./saved_models_bkup/vert_filt_conv_mnist_2.h5",
]
time_conv_mnist_list = [3.54, 2.13, 1]#[37807, 22733, 10681]#[62, 50, 33]
time2_conv_mnist_list = [12.53, 4.54, 1]#[37807, 22733, 10681]#[62, 50, 33]

load_squeeze_cifar10_list = [
	"./vert_filt_saved_keras_models/SqueezeNet_cifar10_false_1.h5",
	"./saved_models_bkup/vert_filt_squeezenet_cifar10_2.h5",
	"./saved_models_bkup/vert_filt_squeezenet_cifar10_3.h5",
]
time_squeeze_cifar10_list = [1.44, 1.28, 1.0]#[3962, 3524, 2751]
time2_squeeze_cifar10_list = [2.08, 1.64, 1.0]#[3962, 3524, 2751]

load_squeeze_mnist_list = [
	"./vert_filt_saved_keras_models/SqueezeNet_mnist_false_1.h5",
	"./saved_models_bkup/vert_filt_squeezenet_mnist_2_12_1.h5",
	"./saved_models_bkup/vert_filt_squeezenet_mnist_3.h5",
]
time_squeeze_mnist_list = [1.61,1.272, 1.0]#[4377, 3456, 2716]
time2_squeeze_mnist_list = [2.59,1.62, 1.0]#[4377, 3456, 2716]

load_mobile_cifar10_list = [
	"./vert_filt_saved_keras_models/MobileNet_cifar10_false_4.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_cifar10_3.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_cifar10_3.0.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_cifar10_2.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_cifar10_2.0.h5",
]
time_mobile_cifar10_list = [3.40, 2.66, 1.95, 1.45, 1.0]#[7823, 6127, 4490, 3330, 2304]
time2_mobile_cifar10_list = [11.56, 7.08, 3.80, 2.10, 1.0]#

load_mobile_mnist_list = [
	"./vert_filt_saved_keras_models/MobileNet_mnist_false_20112018_4.h5", 
	"./vert_filt_saved_models/vert_filt_MobileNet_mnist_3.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_mnist_3.0.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_mnist_2.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_mnist_2.0.h5",
]
time_mobile_mnist_list = [3.17, 2.57, 1.90, 1.37, 1.0]#[7538, 6119, 4514, 3267, 2381]
time2_mobile_mnist_list = [10.1, 6.61, 3.61, 1.88, 1.0]#


load_mobile_cifar10_list_3 = [
	"./vert_filt_saved_keras_models/MobileNet_cifar10_false_4.h5",
	# "./saved_models/vert_filt_MobileNet_cifar10_3.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_cifar10_3.0.h5",
	# "./saved_models/vert_filt_MobileNet_cifar10_2.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_cifar10_2.0.h5",
]
time_mobile_cifar10_list_3 = [3.40, 1.95, 1.0]#[7823, 6127, 4490, 3330, 2304]
time2_mobile_cifar10_list_3 = [11.56, 3.80, 1.0]#

load_mobile_mnist_list_3 = [
	"./vert_filt_saved_keras_models/MobileNet_mnist_false_20112018_4.h5", 
	# "./saved_models/vert_filt_MobileNet_mnist_3.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_mnist_3.0.h5",
	# "./saved_models/vert_filt_MobileNet_mnist_2.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_mnist_2.0.h5",
]
time_mobile_mnist_list_3 = [3.17, 1.90, 1.0]#[7538, 6119, 4514, 3267, 2381]
time2_mobile_mnist_list_3 = [10.1, 3.61, 1.0]#


load_mobile_cifar10_list_4 = [
	"./vert_filt_saved_keras_models/MobileNet_cifar10_false_4.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_cifar10_3.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_cifar10_3.0.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_cifar10_2.5.h5",
	# "./saved_models/vert_filt_MobileNet_cifar10_2.0.h5",
]
time_mobile_cifar10_list_4 = [3.40, 2.66, 1.95, 1.45]#[7823, 6127, 4490, 3330, 2304]
time2_mobile_cifar10_list_4 = [11.56, 7.08, 3.80, 2.10]#

load_mobile_mnist_list_4 = [
	"./vert_filt_saved_keras_models/MobileNet_mnist_false_20112018_4.h5", 
	"./vert_filt_saved_models/vert_filt_MobileNet_mnist_3.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_mnist_3.0.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_mnist_2.5.h5",
	# "./saved_models/vert_filt_MobileNet_mnist_2.0.h5",
]
time_mobile_mnist_list_4 = [3.17, 2.57, 1.90, 1.37]#[7538, 6119, 4514, 3267, 2381]
time2_mobile_mnist_list_4 = [10.1, 6.61, 3.61, 1.88]#

load_mobile_cifar10_list_2 = [
	"./vert_filt_saved_keras_models/MobileNet_cifar10_false_4.h5",
	# "./saved_models/vert_filt_MobileNet_cifar10_3.5.h5",
	# "./saved_models/vert_filt_MobileNet_cifar10_3.0.h5",
	# "./saved_models/vert_filt_MobileNet_cifar10_2.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_cifar10_2.0.h5",
]
time_mobile_cifar10_list_2 = [3.40, 1.0]#[7823, 6127, 4490, 3330, 2304]
time2_mobile_cifar10_list_2 = [11.56, 1.0]#

load_mobile_mnist_list_2 = [
	"./vert_filt_saved_keras_models/MobileNet_mnist_false_20112018_4.h5", 
	# "./saved_models/vert_filt_MobileNet_mnist_3.5.h5",
	# "./saved_models/vert_filt_MobileNet_mnist_3.0.h5",
	# "./saved_models/vert_filt_MobileNet_mnist_2.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_mnist_2.0.h5",
]
time_mobile_mnist_list_2 = [3.17, 1.0]#[7538, 6119, 4514, 3267, 2381]
time2_mobile_mnist_list_2 = [10.1, 1.0]#

load_mobile_cifar10_list_41 = [
	"./vert_filt_saved_keras_models/MobileNet_cifar10_false_4.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_cifar10_3.5.h5",
	# "./vert_filt_saved_models/vert_filt_MobileNet_cifar10_3.0.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_cifar10_2.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_cifar10_2.0.h5",
]
time_mobile_cifar10_list_41 = [3.40, 2.66, 1.45, 1.0]#[7823, 6127, 4490, 3330, 2304]
time2_mobile_cifar10_list_41 = [11.56, 7.08, 2.10, 1.0]#

load_mobile_mnist_list_41 = [
	"./vert_filt_saved_keras_models/MobileNet_mnist_false_20112018_4.h5", 
	"./vert_filt_saved_models/vert_filt_MobileNet_mnist_3.5.h5",
	# "./vert_filt_saved_models/vert_filt_MobileNet_mnist_3.0.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_mnist_2.5.h5",
	"./vert_filt_saved_models/vert_filt_MobileNet_mnist_2.0.h5",
]
time_mobile_mnist_list_41 = [3.17, 2.57, 1.37, 1.0]#[7538, 6119, 4514, 3267, 2381]
time2_mobile_mnist_list_41 = [10.1, 6.61, 1.88, 1.0]#



# config_list = [
# #0
# [load_conv_cifar10_list, time_conv_cifar10_list],
# #1
# [load_cifar10_vgg_list, time_cifar10_vgg_list],
# #2
# [load_conv_mnist_list, time_conv_mnist_list],
# #3
# [load_squeeze_cifar10_list, time_squeeze_cifar10_list],
# #4
# [load_squeeze_mnist_list, time_squeeze_mnist_list],
# ]

config_list = [
#0
[load_conv_cifar10_list, time2_conv_cifar10_list],
#1
[load_cifar10_vgg_list, time2_cifar10_vgg_list],
#2
[load_conv_mnist_list, time2_conv_mnist_list],
#3
[load_squeeze_cifar10_list, time2_squeeze_cifar10_list],
#4
[load_squeeze_mnist_list, time2_squeeze_mnist_list],
#5
[load_mobile_cifar10_list_3, time2_mobile_cifar10_list_3],
#6
[load_mobile_mnist_list_3, time2_mobile_mnist_list_3],
#7
[load_mobile_cifar10_list_4, time2_mobile_cifar10_list_4],
#8
[load_mobile_mnist_list_4, time2_mobile_mnist_list_4],
#9
[load_mobile_cifar10_list_2, time2_mobile_cifar10_list_2],
#10
[load_mobile_mnist_list_2, time2_mobile_mnist_list_2],
#11
[load_mobile_cifar10_list, time2_mobile_cifar10_list],
#12
[load_mobile_mnist_list, time2_mobile_mnist_list],
#13
[load_mobile_cifar10_list_41, time2_mobile_cifar10_list_41],
#14
[load_mobile_mnist_list_41, time2_mobile_mnist_list_41],
]
BASE_FOLDER = "/mnt/additional/nitthilan/ml_tutorials/conv/"


config_list = [
#0
[
["vert_filt_conv_svhn_2.0", "vert_filt_conv_svhn_2.5", "vert_filt_conv_svhn_3.0", "vert_filt_conv_svhn_3.5", "vert_filt_conv_svhn_4.0"],
[104.422, 241.97, 439.875, 818.652, 1267.29]
],
#1
[
["vert_filt_conv_mnist_2.0", "vert_filt_conv_mnist_2.5", "vert_filt_conv_mnist_3.0", "vert_filt_conv_mnist_3.5", "vert_filt_conv_mnist_4.0"],
[102.13, 258.914, 457.177, 852.973, 1322.297]
],
#2
[
["vert_filt_conv_cifar10_2.25", "vert_filt_conv_cifar10_2.5", "vert_filt_conv_cifar10_3.0", "vert_filt_conv_cifar10_3.5", "vert_filt_conv_cifar10_4.0"],
[167.23, 241.865, 435.386, 821.648, 1263.113]
],

#3
[
["vert_filt_vgg_svhn_2.0", "vert_filt_vgg_svhn_2.5", "vert_filt_vgg_svhn_3.0", "vert_filt_vgg_svhn_3.5", "vert_filt_vgg_svhn_4.0"],
[495.018, 1197.022, 2297.63, 4234.26, 7149.29]
],
#4
[
["vert_filt_vgg_mnist_2.0", "vert_filt_vgg_mnist_2.5", "vert_filt_vgg_mnist_3.0", "vert_filt_vgg_mnist_3.5", "vert_filt_vgg_mnist_4.0"],
[482.19, 1218.21, 2352.33, 4325.84, 7350.06]
],
#5
[
["vert_filt_vgg_cifar10_2.25", "vert_filt_vgg_cifar10_2.5", "vert_filt_vgg_cifar10_3.0", "vert_filt_vgg_cifar10_3.5", "vert_filt_vgg_cifar10_4.0"],
[793.75, 1197.499, 2288.178, 4237.65, 7120.997]
],

#6
[
["vert_filt_MobileNet_svhn_2.0", "vert_filt_MobileNet_svhn_2.5", "vert_filt_MobileNet_svhn_3.0", "vert_filt_MobileNet_svhn_3.5", "vert_filt_MobileNet_svhn_4.0"],
[6.633, 12.342, 25.463, 43.376, 67.497]
],
#7
[
["vert_filt_MobileNet_mnist_2.0", "vert_filt_MobileNet_mnist_2.5", "vert_filt_MobileNet_mnist_3.0", "vert_filt_MobileNet_mnist_3.5", "vert_filt_MobileNet_mnist_4.0"],
[6.53, 12.62, 23.996, 43.47, 66.798]
],
#8
[
["vert_filt_MobileNet_cifar10_2.25", "vert_filt_MobileNet_cifar10_2.5", "vert_filt_MobileNet_cifar10_3.0", "vert_filt_MobileNet_cifar10_3.5", "vert_filt_MobileNet_cifar10_4.0"],
[8.97, 12.0, 24.09, 41.81, 67.29]
],

#9
[
["VGG190.5", "VGG190.625", "VGG190.75", "VGG190.875", "VGG191.0"],
[793.75, 1197.499, 2288.178, 4237.65, 7120.997]
],

#10
[
["MobileNet0.5", "MobileNet0.625", "MobileNet0.75", "MobileNet0.875", "MobileNet1.0"],
# [52.961, 76.949, 105.646, 139.937, 173.826]
[2804.9, 5921.2, 11161.1, 19582.4, 30215.5]

],
]

for config in config_list:
	for array in config:
		array.reverse()

print(config_list[0])

BASE_FOLDER = "/mnt/additional/nitthilan/ml_tutorials/conv/vert_filt_models/"

BASE_FOLDER = "/mnt/additional/nitthilan/data/ml_tutorial/imagenet/"
